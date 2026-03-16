close all
clear
clc

%% Load ADCP %%

file_in='MAVS2_24606.nc';

ADCP.time=(double(ncread(file_in,'time'))./1e6./60./60./24)+datenum(2021,7,7,6,15,0);
ADCP.z=double(ncread(file_in,'z'));
ADCP.u=double(ncread(file_in,'u'));
ADCP.v=double(ncread(file_in,'v'));
ADCP.w=double(ncread(file_in,'w'));

ADCP.z=ADCP.z(2:17);
ADCP.u=ADCP.u(:,2:17);
ADCP.v=ADCP.v(:,2:17);
ADCP.w=ADCP.w(:,2:17);

ADCP.time_tmp=ADCP.time;
ADCP.time=interp1(1:length(ADCP.time),ADCP.time,1:1/5:length(ADCP.time));

for ii=1:length(ADCP.z)
    ADCP.u_tmp(:,ii)=interp1(ADCP.time_tmp,ADCP.u(:,ii),ADCP.time);
    ADCP.v_tmp(:,ii)=interp1(ADCP.time_tmp,ADCP.v(:,ii),ADCP.time);
    ADCP.w_tmp(:,ii)=interp1(ADCP.time_tmp,ADCP.w(:,ii),ADCP.time);
end

ADCP.u=ADCP.u_tmp;
ADCP.v=ADCP.v_tmp;
ADCP.w=ADCP.w_tmp;

clear ADCP.u_tmp ADCP.v_tmp ADCP.w_tmp ADCP.time_tmp

%% Load T Chain - UnCal%%

path='./mavs2_l1';

% file_dates=datenum(2021,7,6):2:datenum(2021,10,04);
file_dates=datenum(2021,7,6):2:datenum(2021,7,20);

TChain.temp=[];
TChain.time=[];

for ii=1:length(file_dates)
    
    disp(['Doing ',num2str(ii),' of ',num2str(length(file_dates))])
    
    filename=[path,'/mavs2_',datestr(file_dates(ii),'yyyymmdd'),'.nc'];
    
    time=double(ncread(filename,'time'))./60./60./24+file_dates(ii);
    depth=ncread(filename,'depth');
    temp=ncread(filename,'__xarray_dataarray_variable__');
    
    TChain.time=[TChain.time time'];
    
    TChain.temp=[TChain.temp temp'];
    
    TChain.depth=depth;
    
    clear time depth temp
    
end

%% Load T Chain - Cal%%

path_cal='./mavs2_l2';

% file_dates=datenum(2021,7,6):2:datenum(2021,10,04);
file_dates=datenum(2021,7,6):2:datenum(2021,7,20);

TChain_cal.temp=[];
TChain_cal.time=[];

for ii=1:length(file_dates)
    
    disp(['Doing ',num2str(ii),' of ',num2str(length(file_dates))])
    
    filename=[path_cal,'/mavs2_',datestr(file_dates(ii),'yyyymmdd'),'.nc'];
    
    time=double(ncread(filename,'time'))./60./60./24+file_dates(ii);
    depth=ncread(filename,'depth');
    temp=ncread(filename,'__xarray_dataarray_variable__');
    
    TChain_cal.time=[TChain_cal.time time'];
    
    TChain_cal.temp=[TChain_cal.temp temp'];
    
    TChain_cal.depth=depth;
    
    clear time depth temp
    
end

%% Pass Parts To A Function %%

chi_time_step=10./60./24; % in seconds

chi.time_bnds=TChain.time(1):chi_time_step:TChain.time(end);

chi.depth=TChain.depth;

chi_spectra_bin_bounds=-12:1:-3; % sets the averaging bins for plots of spectra

chi.chi=nan([length(chi.depth) length(chi.time_bnds)-1]);

ft=fittype('a+b*x');

for ii=1:length(chi.depth)
    for kk=1:length(chi_spectra_bin_bounds)-1

        avrg_spectra(ii).spectra(kk).sum=[];
        avrg_spectra(ii).spectra(kk).count=0;

    end
end

for ii=1:length(chi.depth) 
    for jj=1:length(chi.time_bnds)-1 % Run through all depths and chunk in time

        tic
        
        disp(['Doing ',num2str(ii),' of ',num2str(length(chi.depth)),' and ',num2str(jj),' of ',num2str(length(chi.time_bnds)-1)])
        
        chi.time(jj)=(chi.time_bnds(jj)+chi.time_bnds(jj+1))./2;

        tmp_indx=find(TChain_cal.time >= chi.time_bnds(jj) & TChain_cal.time < chi.time_bnds(jj+1)); % find time index for section

        temp_in=TChain.temp(ii,tmp_indx);

        temp_in_cal=TChain_cal.temp(ii,tmp_indx);

        z_indx=find(abs(ADCP.z-chi.depth(ii)) == min(abs(ADCP.z-chi.depth(ii)))); % finds the closest ADCP bin

        U_in=sqrt(nanmean(ADCP.u(ADCP.time >= chi.time_bnds(jj) & ADCP.time < chi.time_bnds(jj+1),z_indx)).^2+nanmean(ADCP.v(ADCP.time >= chi.time_bnds(jj) & ADCP.time < chi.time_bnds(jj+1),z_indx)).^2);
        
        if ii==1 % calc vertical temp grads

            dtdz=(nanmean(TChain_cal.temp(ii,tmp_indx))-nanmean(TChain_cal.temp(ii+1,tmp_indx)))./(chi.depth(ii)-chi.depth(ii+1));
            dtdz_ts=((TChain_cal.temp(ii,tmp_indx))-(TChain_cal.temp(ii+1,tmp_indx)))./(chi.depth(ii)-chi.depth(ii+1));

        elseif ii==length(chi.depth)
            
            dtdz=(nanmean(TChain_cal.temp(ii-1,tmp_indx))-nanmean(TChain_cal.temp(ii,tmp_indx)))./(chi.depth(ii-1)-chi.depth(ii));
            dtdz_ts=((TChain_cal.temp(ii-1,tmp_indx))-(TChain_cal.temp(ii,tmp_indx)))./(chi.depth(ii-1)-chi.depth(ii));

        else
            
            dtdz=(nanmean(TChain_cal.temp(ii-1,tmp_indx))-nanmean(TChain_cal.temp(ii+1,tmp_indx)))./(chi.depth(ii-1)-chi.depth(ii+1));
            dtdz_ts=((TChain_cal.temp(ii-1,tmp_indx))-(TChain_cal.temp(ii+1,tmp_indx)))./(chi.depth(ii-1)-chi.depth(ii+1));

        end

        dtdx=((temp_in_cal(end)-temp_in_cal(1))./(chi_time_step.*24.*60.*60))./U_in; % estiamte dtdx using frozen field
        
        unstab_count=length(find(dtdz_ts > 0));
        unstab_length=length(find(~isnan(dtdz_ts)));
        unstab_prop=unstab_count./unstab_length;
        
        clear dtdz_ts
        
        gamma=0.2; % make an assumption about mixing eff
        
        alpha=sw_alpha(35,nanmean(TChain_cal.temp(ii,tmp_indx)),sw_pres(chi.depth(ii),54+(15/60)));
        
        if length(temp_in) == length(temp_in(~isnan(temp_in)))
            
            spectra_size=2^7; % set the size of the spectra calculated to then be averaged in Welch (might need to change for different sampling rate or averaging limits)

            avrg_lim(1)=0.8e-2; % averaging limits in Hz for the upper and lower bounds of the interial subrange
            avrg_lim(2)=1e-1;

            U_tmp=0.1;  % some representative value of the typical flow in the region
            hab_tmp=1466-chi.depth(ii);   % take height above bottom

            u_h=U_tmp./hab_tmp;   % set u/h scale to limit the low freq end of the averaging to account for turb eddies being supressed by boundary
            
            if u_h>avrg_lim(1); avrg_lim(1)=u_h; end

            [chi.chi(ii,jj),diag]=Calc_Chi_TChain_2(temp_in,U_in,gamma,alpha,sqrt(dtdz.^2+dtdx.^2),0,spectra_size,avrg_lim);  % pass everything to the calculation
            
            %saveas(gcf,['./Spectra/Spectra_',num2str(ii),',',num2str(jj),'.eps'],'epsc')

            kk=find(chi_spectra_bin_bounds<log10(chi.chi(ii,jj)) & chi_spectra_bin_bounds+1>=log10(chi.chi(ii,jj)));  %% find the spectra for the chi bins to average for plotting

            if ~isempty(kk)
                if isempty(avrg_spectra(ii).spectra(kk).sum)

                    avrg_spectra(ii).spectra(kk).sum=diag.Pt;

                    avrg_spectra(ii).spectra(kk).count=1;

                else

                    avrg_spectra(ii).spectra(kk).sum=avrg_spectra(ii).spectra(kk).sum+diag.Pt;

                    avrg_spectra(ii).spectra(kk).count=avrg_spectra(ii).spectra(kk).count+1;

                end
            end

            indx=find(diag.f>avrg_lim(1)./1.5 & diag.f<avrg_lim(2));

            if ~isempty(find(diag.Pt(indx)~=0))

                fitobj=fit(log10(diag.f(indx)),log10(diag.Pt(indx)),ft,'StartPoint',[1,-5/3]);

                avrg_spectra(ii).slope(jj)=fitobj.b;
                
                % fh=figure('units','normalized','outerposition',[0 0 .5 .5]);
                % plot(log10(diag.f),log10(diag.Pt))
                % xlabel('Frequency')
                % %set(gca,'XScale','log','YScale','log')
                % hold on
                % plot([1 1].*log10(avrg_lim(1)),ylim,'k')
                % plot([1 1].*log10(avrg_lim(2)),ylim,'k')
                % plot(fitobj)
                % title(num2str(fitobj.b))
                % 
                % pause
                % 
                % close(fh)

            else

                avrg_spectra(ii).slope(jj)=nan;

            end
            
            % Tidy everything up

            avrg_spectra(ii).avrg_lim=avrg_lim; 

            avrg_spectra(ii).f=diag.f;

            chi.U(ii,jj)=U_in;
            
            chi.mean_u(ii,jj)=interp1(ADCP.z,nanmean(ADCP.u(ADCP.time >= chi.time_bnds(jj) & ADCP.time < chi.time_bnds(jj+1),:),1),chi.depth(ii));
            
            chi.mean_v(ii,jj)=interp1(ADCP.z,nanmean(ADCP.v(ADCP.time >= chi.time_bnds(jj) & ADCP.time < chi.time_bnds(jj+1),:),1),chi.depth(ii));
            
            chi.mean_w(ii,jj)=interp1(ADCP.z,nanmean(ADCP.w(ADCP.time >= chi.time_bnds(jj) & ADCP.time < chi.time_bnds(jj+1),:),1),chi.depth(ii));
            
            chi.mean_t_uncal(ii,jj)=diag.mean_t;
            
            chi.mean_t(ii,jj)=nanmean(TChain_cal.temp(ii,tmp_indx));
            
            chi.dtdz(ii,jj)=dtdz;

            chi.dtdx(ii,jj)=dtdx;
            
            chi.unstab_count(ii,jj)=unstab_count;
            
            chi.unstab_length(ii,jj)=unstab_length;
            
            chi.unstab_prop(ii,jj)=unstab_prop;
            
            chi.alpha(ii,jj)=alpha;
            
            chi.gamma(ii,jj)=gamma;
            
        else
            
            chi.chi(ii,jj)=nan;
            
            chi.U(ii,jj)=U_in;
            
            chi.mean_u(ii,jj)=nanmean(ADCP.u(ADCP.time >= chi.time_bnds(jj) & ADCP.time < chi.time_bnds(jj+1),z_indx));
            
            chi.mean_v(ii,jj)=nanmean(ADCP.v(ADCP.time >= chi.time_bnds(jj) & ADCP.time < chi.time_bnds(jj+1),z_indx));
            
            chi.mean_w(ii,jj)=nanmean(ADCP.w(ADCP.time >= chi.time_bnds(jj) & ADCP.time < chi.time_bnds(jj+1),z_indx));
            
            chi.mean_t_uncal(ii,jj)=nan;
            
            chi.mean_t(ii,jj)=nanmean(TChain_cal.temp(ii,TChain_cal.time >= chi.time_bnds(jj) & TChain_cal.time < chi.time_bnds(jj+1)));
            
            chi.dtdz(ii,jj)=dtdz;
            
            chi.dtdx(ii,jj)=dtdx;

            chi.unstab_count(ii,jj)=unstab_count;
            
            chi.unstab_length(ii,jj)=unstab_length;
            
            chi.unstab_prop(ii,jj)=unstab_prop;
            
            chi.alpha(ii,jj)=alpha;
            
            chi.gamma(ii,jj)=nan;

            avrg_spectra(ii).slope(jj)=nan;
            
        end

        toc
        
    end
end

for ii=1:length(chi.depth)
    for kk=1:length(chi_spectra_bin_bounds)-1

        avrg_spectra(ii).spectra(kk).mean=avrg_spectra(ii).spectra(kk).sum./avrg_spectra(ii).spectra(kk).count;

    end
end

%% Plot Averaged Spectra and histogram of slopes for each thermistor %%


for ii=1:length(chi.depth)

    fh=figure;

    for kk=1:length(chi_spectra_bin_bounds)-1

        if ~isempty(avrg_spectra(ii).spectra(kk).sum)

            loglog(avrg_spectra(ii).f,avrg_spectra(ii).spectra(kk).mean,'k','LineWidth',1.5);
            hold on
        end

    end

    loglog(avrg_spectra(ii).avrg_lim(1).*[1 1],ylim,'r:','LineWidth',1.5)
    loglog(avrg_spectra(ii).avrg_lim(2).*[1 1],ylim,'r:','LineWidth',1.5)
    loglog([1e-2 1e-1],0.5e-8.*[1e-2 1e-1].^(-5/3),'g--','LineWidth',1.5)
    loglog([1e-2 1e-1],0.5e-6.*[1e-2 1e-1].^(-5/3),'g--','LineWidth',1.5)
    loglog([1e-2 1e-1],0.5e-4.*[1e-2 1e-1].^(-5/3),'g--','LineWidth',1.5)
    title(['Average Spectra at ',num2str(chi.depth(ii)),'m'])

    saveas(fh,['./Spectra_Plots/Spectra_',num2str(ii),'.fig']);
    saveas(fh,['./Spectra_Plots/Spectra_',num2str(ii),'.png']);

    close(fh)

end

for ii=1:length(chi.depth)

    fh=figure;

    histogram(avrg_spectra(ii).slope,-4:0.05:0)
    hold on
    plot(-5/3.*[1 1],ylim,'r','LineWidth',1.5)
    title(['Histogram of spectral slope at ',num2str(chi.depth(ii)),'m'])

    saveas(fh,['./Spectra_Plots/Slope_Hist_',num2str(ii),'.fig']);
    saveas(fh,['./Spectra_Plots/Slope_Hist_',num2str(ii),'.png']);

    close(fh)

end

%% Save Output %%

save('TChain_chi_10mins_Horiz_Grads_hu.mat','-struct','chi');

save('TChain_chi_Spectra.mat','avrg_spectra','chi_spectra_bin_bounds');
