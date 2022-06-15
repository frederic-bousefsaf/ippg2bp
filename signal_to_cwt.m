function [CWT] = signal_to_cwt(signal, overlap, norm, detrend, fps)
% signal: full iPPG or BP signal (sampling frequency=fps)
% overlap: 0 for no overlap; N for an overlap on N samples
% norm: 0 for no standardization (BP); 1 for standardization (iPPG)
% detrend: 0 for no detrending (BP) 1 for detrending (iPPG)
% fps: sampling frequency of the signal

%% COMPUTE SCALES
sc_min = -1;
sc_max = -1;
sc = 0.2:0.01:1000;
MorletFourierFactor = 4*pi/(6+sqrt(2+6^2));
freqs = 1./(sc.*MorletFourierFactor);
for dummy=1:length(freqs)
    if (freqs(dummy)<0.6 && sc_max==-1)     % can be adjusted
        sc_max = sc(dummy);
        
    elseif (freqs(dummy)<8 && sc_min==-1)   % can be adjusted
        sc_min = sc(dummy);
    end
end
sc = [sc_min sc_max];


%% RESAMPLING (100 Hz)
time = 0:1/100:(length(signal)/fps)-1/fps;
signal = interp1(0:1/fps:(length(signal)/fps)-1/fps, signal, time);
fps = 100;


%% DETRENDING (Tarvainen et al., 2002)
if (detrend)
    lambda = 470;
    T = length(signal);
    I = speye(T);
    D2 = spdiags(ones(T-2,1)*[1 -2 1],[0:2],T-2,T);
    signal = (I-inv(I+lambda^2*D2'*D2))*signal';
end


%% OVERLAPPING
if (overlap == 0)
    overlap = 256;
end
    

%% WINDOWING
i = 1;
c = 1;
while ((i+255)<=length(signal))
    signal_window = signal(i:i+255);
    time_window = time(i:i+255);

    % Standardization
    if (norm)
        signal_window = (signal_window - mean(signal_window)) / std(signal_window);
    end
       
    % Compute CWT
    cwA = cwtft({signal_window,1/fps},'scales',{sc(1) 0.00555 ceil((sc(2)-sc(1))/0.00555) 'lin'},'wavelet','morl');
    CWT{c} = cwA;

    i = i + overlap;
    c = c + 1;
end


end