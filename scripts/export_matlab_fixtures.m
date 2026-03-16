%% export_matlab_fixtures.m
% Requires: seawater toolbox (sw_alpha, sw_pres) on Matlab path.
% Generates test fixtures for pychi by running the Matlab chi pipeline
% on a small data subset and saving intermediate results at each stage.
%
% Run from the matlab_version/ directory after pointing paths to data.
% Saves .mat files to ../tests/fixtures/

output_dir = '../tests/fixtures';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Load ADCP
file_in = 'MAVS2_24606.nc';
ADCP.time = (double(ncread(file_in,'time'))./1e6./60./60./24) + datenum(2021,7,7,6,15,0);
ADCP.z = double(ncread(file_in,'z'));
ADCP.u = double(ncread(file_in,'u'));
ADCP.v = double(ncread(file_in,'v'));
ADCP.w = double(ncread(file_in,'w'));

ADCP.z = ADCP.z(2:17);
ADCP.u = ADCP.u(:,2:17);
ADCP.v = ADCP.v(:,2:17);
ADCP.w = ADCP.w(:,2:17);

ADCP.time_tmp = ADCP.time;
ADCP.time = interp1(1:length(ADCP.time), ADCP.time, 1:1/5:length(ADCP.time));
for ii = 1:length(ADCP.z)
    ADCP.u_tmp(:,ii) = interp1(ADCP.time_tmp, ADCP.u(:,ii), ADCP.time);
    ADCP.v_tmp(:,ii) = interp1(ADCP.time_tmp, ADCP.v(:,ii), ADCP.time);
    ADCP.w_tmp(:,ii) = interp1(ADCP.time_tmp, ADCP.w(:,ii), ADCP.time);
end
ADCP.u = ADCP.u_tmp;
ADCP.v = ADCP.v_tmp;
ADCP.w = ADCP.w_tmp;

%% Load T-chain (uncalibrated L1)
path_l1 = './mavs2_l1';
file_dates = datenum(2021,7,10):2:datenum(2021,7,12);  % Small subset: 2 files
TChain.temp = []; TChain.time = [];
for ii = 1:length(file_dates)
    filename = [path_l1, '/mavs2_', datestr(file_dates(ii),'yyyymmdd'), '.nc'];
    time = double(ncread(filename,'time'))./60./60./24 + file_dates(ii);
    depth = ncread(filename,'depth');
    temp = ncread(filename,'__xarray_dataarray_variable__');
    TChain.time = [TChain.time time'];
    TChain.temp = [TChain.temp temp'];
    TChain.depth = depth;
end

%% Load T-chain (calibrated L2)
path_l2 = './mavs2_l2';
TChain_cal.temp = []; TChain_cal.time = [];
for ii = 1:length(file_dates)
    filename = [path_l2, '/mavs2_', datestr(file_dates(ii),'yyyymmdd'), '.nc'];
    time = double(ncread(filename,'time'))./60./60./24 + file_dates(ii);
    depth = ncread(filename,'depth');
    temp = ncread(filename,'__xarray_dataarray_variable__');
    TChain_cal.time = [TChain_cal.time time'];
    TChain_cal.temp = [TChain_cal.temp temp'];
    TChain_cal.depth = depth;
end

%% Parameters
chi_time_step = 10/60/24;
spectra_size = 2^7;
sample_freq = 1;
gamma = 0.2;

chi.time_bnds = TChain.time(1):chi_time_step:TChain.time(end);
chi.depth = TChain.depth;

%% Select fixture chunks
fixture_cases = [1, 1;
                 length(chi.depth), 1;
                 round(length(chi.depth)/2), round(length(chi.time_bnds)/4);
                 round(length(chi.depth)/2), round(length(chi.time_bnds)/2);
                 1, 2];

for cc = 1:size(fixture_cases, 1)
    ii = fixture_cases(cc, 1);
    jj = fixture_cases(cc, 2);

    if jj >= length(chi.time_bnds), continue; end

    tmp_indx = find(TChain_cal.time >= chi.time_bnds(jj) & TChain_cal.time < chi.time_bnds(jj+1));
    if isempty(tmp_indx), continue; end

    temp_in = TChain.temp(ii, tmp_indx);
    temp_in_cal = TChain_cal.temp(ii, tmp_indx);

    z_indx = find(abs(ADCP.z - chi.depth(ii)) == min(abs(ADCP.z - chi.depth(ii))));
    U_in = sqrt(nanmean(ADCP.u(ADCP.time >= chi.time_bnds(jj) & ADCP.time < chi.time_bnds(jj+1), z_indx)).^2 + ...
                nanmean(ADCP.v(ADCP.time >= chi.time_bnds(jj) & ADCP.time < chi.time_bnds(jj+1), z_indx)).^2);

    if ii == 1
        dtdz = (nanmean(TChain_cal.temp(ii,tmp_indx)) - nanmean(TChain_cal.temp(ii+1,tmp_indx))) ./ (chi.depth(ii) - chi.depth(ii+1));
        dtdz_ts = (TChain_cal.temp(ii,tmp_indx) - TChain_cal.temp(ii+1,tmp_indx)) ./ (chi.depth(ii) - chi.depth(ii+1));
    elseif ii == length(chi.depth)
        dtdz = (nanmean(TChain_cal.temp(ii-1,tmp_indx)) - nanmean(TChain_cal.temp(ii,tmp_indx))) ./ (chi.depth(ii-1) - chi.depth(ii));
        dtdz_ts = (TChain_cal.temp(ii-1,tmp_indx) - TChain_cal.temp(ii,tmp_indx)) ./ (chi.depth(ii-1) - chi.depth(ii));
    else
        dtdz = (nanmean(TChain_cal.temp(ii-1,tmp_indx)) - nanmean(TChain_cal.temp(ii+1,tmp_indx))) ./ (chi.depth(ii-1) - chi.depth(ii+1));
        dtdz_ts = (TChain_cal.temp(ii-1,tmp_indx) - TChain_cal.temp(ii+1,tmp_indx)) ./ (chi.depth(ii-1) - chi.depth(ii+1));
    end

    dtdx = ((temp_in_cal(end) - temp_in_cal(1)) ./ (chi_time_step.*24.*60.*60)) ./ U_in;

    alpha_val = sw_alpha(35, nanmean(TChain_cal.temp(ii,tmp_indx)), sw_pres(chi.depth(ii), 54+(15/60)));

    avrg_lim = [0.8e-2, 1e-1];
    U_tmp = 0.1;
    hab_tmp = 1466 - chi.depth(ii);
    u_h = U_tmp / hab_tmp;
    if u_h > avrg_lim(1); avrg_lim(1) = u_h; end

    unstab_count = length(find(dtdz_ts > 0));
    unstab_length = length(find(~isnan(dtdz_ts)));

    has_nan = length(temp_in) ~= length(temp_in(~isnan(temp_in)));

    if ~has_nan
        win = hanning(spectra_size) ./ sqrt(mean(hanning(spectra_size).^2));
        [Pt, f] = csd_odas(temp_in, temp_in, spectra_size, sample_freq, win, spectra_size/2, 'linear');

        grad_T_mag = sqrt(dtdz.^2 + dtdx.^2);
        [chi_val, diag_out] = Calc_Chi_TChain_2(temp_in, U_in, gamma, alpha_val, grad_T_mag, 0, spectra_size, avrg_lim);

        save(fullfile(output_dir, ['spectra_chunk_', num2str(cc), '.mat']), ...
            'temp_in', 'Pt', 'f', 'win', 'spectra_size', 'sample_freq', 'avrg_lim');

        save(fullfile(output_dir, ['chi_chunk_', num2str(cc), '.mat']), ...
            'chi_val', 'alpha_val', 'gamma', 'U_in', 'grad_T_mag', ...
            'dtdz', 'dtdx', 'unstab_count', 'unstab_length', ...
            'temp_in', 'temp_in_cal', 'avrg_lim');
    end

    if ii == 1
        temp_cal_neighbors = TChain_cal.temp(1:2, tmp_indx);
        depths_neighbors = chi.depth(1:2);
    elseif ii == length(chi.depth)
        temp_cal_neighbors = TChain_cal.temp(end-1:end, tmp_indx);
        depths_neighbors = chi.depth(end-1:end);
    else
        temp_cal_neighbors = TChain_cal.temp(ii-1:ii+1, tmp_indx);
        depths_neighbors = chi.depth(ii-1:ii+1);
    end

    save(fullfile(output_dir, ['gradient_chunk_', num2str(cc), '.mat']), ...
        'temp_cal_neighbors', 'depths_neighbors', 'dtdz', 'dtdx', 'U_in', ...
        'ii', 'has_nan');
end

%% Save small end-to-end fixture (3 depths x 10 chunks)
n_test_depths = min(3, length(chi.depth));
n_test_chunks = min(10, length(chi.time_bnds) - 1);

e2e_chi = nan(n_test_depths, n_test_chunks);
e2e_dtdz = nan(n_test_depths, n_test_chunks);
e2e_dtdx = nan(n_test_depths, n_test_chunks);
e2e_U = nan(n_test_depths, n_test_chunks);
e2e_alpha = nan(n_test_depths, n_test_chunks);
e2e_slope = nan(n_test_depths, n_test_chunks);

ft = fittype('a+b*x');

for ii = 1:n_test_depths
    for jj = 1:n_test_chunks
        tmp_indx = find(TChain_cal.time >= chi.time_bnds(jj) & TChain_cal.time < chi.time_bnds(jj+1));
        if isempty(tmp_indx), continue; end

        temp_in = TChain.temp(ii, tmp_indx);
        temp_in_cal = TChain_cal.temp(ii, tmp_indx);

        z_indx = find(abs(ADCP.z - chi.depth(ii)) == min(abs(ADCP.z - chi.depth(ii))));
        U_in = sqrt(nanmean(ADCP.u(ADCP.time >= chi.time_bnds(jj) & ADCP.time < chi.time_bnds(jj+1), z_indx)).^2 + ...
                    nanmean(ADCP.v(ADCP.time >= chi.time_bnds(jj) & ADCP.time < chi.time_bnds(jj+1), z_indx)).^2);

        if ii == 1
            dtdz = (nanmean(TChain_cal.temp(ii,tmp_indx)) - nanmean(TChain_cal.temp(ii+1,tmp_indx))) ./ (chi.depth(ii) - chi.depth(ii+1));
        elseif ii == n_test_depths
            dtdz = (nanmean(TChain_cal.temp(ii-1,tmp_indx)) - nanmean(TChain_cal.temp(ii,tmp_indx))) ./ (chi.depth(ii-1) - chi.depth(ii));
        else
            dtdz = (nanmean(TChain_cal.temp(ii-1,tmp_indx)) - nanmean(TChain_cal.temp(ii+1,tmp_indx))) ./ (chi.depth(ii-1) - chi.depth(ii+1));
        end

        dtdx = ((temp_in_cal(end) - temp_in_cal(1)) ./ (chi_time_step.*24.*60.*60)) ./ U_in;
        alpha_val = sw_alpha(35, nanmean(TChain_cal.temp(ii,tmp_indx)), sw_pres(chi.depth(ii), 54+(15/60)));

        avrg_lim = [0.8e-2, 1e-1];
        u_h = 0.1 / (1466 - chi.depth(ii));
        if u_h > avrg_lim(1); avrg_lim(1) = u_h; end

        e2e_dtdz(ii,jj) = dtdz;
        e2e_dtdx(ii,jj) = dtdx;
        e2e_U(ii,jj) = U_in;
        e2e_alpha(ii,jj) = alpha_val;

        if length(temp_in) == length(temp_in(~isnan(temp_in)))
            grad_T_mag = sqrt(dtdz.^2 + dtdx.^2);
            [chi_val, diag_out] = Calc_Chi_TChain_2(temp_in, U_in, gamma, alpha_val, grad_T_mag, 0, spectra_size, avrg_lim);
            e2e_chi(ii,jj) = chi_val;

            indx = find(diag_out.f > avrg_lim(1)/1.5 & diag_out.f < avrg_lim(2));
            if ~isempty(find(diag_out.Pt(indx) ~= 0))
                fitobj = fit(log10(diag_out.f(indx)), log10(diag_out.Pt(indx)), ft, 'StartPoint', [1, -5/3]);
                e2e_slope(ii,jj) = fitobj.b;
            end
        end
    end
end

e2e_depths = chi.depth(1:n_test_depths);
e2e_time_bnds = chi.time_bnds(1:n_test_chunks+1);

save(fullfile(output_dir, 'pipeline_subset.mat'), ...
    'e2e_chi', 'e2e_dtdz', 'e2e_dtdx', 'e2e_U', 'e2e_alpha', 'e2e_slope', ...
    'e2e_depths', 'e2e_time_bnds', 'gamma', 'spectra_size', 'sample_freq');

disp('Fixture export complete.');
