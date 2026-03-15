function [chi,diag] = Calc_Chi_TChain(temperature,U_in,gamma,alpha,dtdz,plot_flag,spectra_size,avrg_lim)
%MAVS_EPS_CHI

%spectra_size=2^7; % This is 128 seconds

sample_freq=1;

%% Do Spectra %%

[Pt,f]=csd_odas(temperature(~isnan(temperature)),temperature(~isnan(temperature)),spectra_size,sample_freq,hanning(spectra_size)./sqrt(mean(hanning(spectra_size).^2)),spectra_size/2,'linear');

scale=(f).^(5/3);

%% Decide limits for averaging %%

%avrg_lim(1)=1e-2;
%avrg_lim(2)=1e-1;

%% Plot up some spectra %%

if plot_flag==1
    
    figure
    loglog(f,Pt,'g','LineWidth',2);
    hold on
    loglog([avrg_lim(1) avrg_lim(1)],ylim,'c')
    loglog([avrg_lim(2) avrg_lim(2)],ylim,'c')
    xlabel('frequency [cps]','FontSize',14)
    legend('t','u1','u2','u3')
    ylabel('spectral density [(m^2 s^{-2}, C^2)s^{-1}]','FontSize',14)
    
    figure
    loglog(f,Pt.*scale,'g','LineWidth',2);
    hold on
    loglog([avrg_lim(1) avrg_lim(1)],ylim,'c')
    loglog([avrg_lim(2) avrg_lim(2)],ylim,'c')
    xlabel('frequency [cps]','FontSize',14)
    legend('t','u1','u2','u3')
    ylabel('spectral density [(m^2 s^{-2}, C^2)s^{-1}]','FontSize',14)
    
end

%% Calculations of chi %%

U=U_in;

g=9.81;

dtdz=abs(dtdz);

chi=(nanmedian(Pt(f>avrg_lim(1) & f<avrg_lim(2)).*scale(f>avrg_lim(1) & f<avrg_lim(2))).*(2.*pi()./U).^(2/3)./0.4.*(g.*alpha./(2.*dtdz.*gamma)).^(1/3)).^(3/2);

%% Tidy Up Outputs %%

diag.Pt=Pt;
diag.f=f;
diag.U=U;
diag.mean_t=nanmean(temperature);


end

