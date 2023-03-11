%mod = comm.DBPSKModulator;
%demod = comm.DBPSKDemodulator;
%chan = comm.RayleighChannel(
 %   SampleRate=1e4,
 %   MaximumDopplerShift=100);

%rayChan = comm.RayleighChannel(
 %   SampleRate=bitRate,
  %  MaximumDopplerShift=4,
   % PathDelays=[0 2e-5],
    %AveragePathGains=[0 -9]);

qpskMod = comm.QPSKModulator;
qpskDemod = comm.QPSKDemodulator;
rayleighchan = comm.RayleighChannel(
    'SampleRate',10e3,
    'PathDelays',[0 1.5e-4],
    'AveragePathGains',[2 3],
    'NormalizePathGains',true,
    'MaximumDopplerShift',30,
    'DopplerSpectrum',{doppler('Gaussian',0.6),doppler('Flat')},
    'RandomStream','mt19937ar with seed',
    'Seed',22,
    'PathGainsOutputPort',true);
awgnChan = comm.AWGNChannel(
    NoiseMethod='Signal to noise ratio (SNR)');
errorCalc = comm.ErrorRate;
bitRate = 20e3;


%bitRate = 1e4;

M = 4;                       % DBPSK modulation order
%tx = randi([0 M-1],50000,1); % Generate a random bit stream
tx = randi([0 M-1],100000,1);
%tx = randi()
%disp(tx)
%disp('*********************************************')

%dpskSig = mod(tx);
%fadedSig = chan(dpskSig);

qpskSig = qpskMod(tx);
fadedSig = rayleighchan(qpskSig);
disp(fadedSig)

SNR = 0:2:20;
numSNR = length(SNR);
berVec = zeros(3, numSNR);

for n = 1:numSNR
   awgnChan.SNR = SNR(n);
   rxSig = awgnChan(fadedSig);
   rx = qpskDemod(rxSig);
   reset(errorCalc)

   berVec(:,n) = errorCalc(tx,rx);
end
BER = berVec(1,:);
%disp(BER)

BERtheory = berfading(SNR,'oqpsk',M,1);
%disp('*********************************************')
%disp(SNR)
%disp('*********************************************')
%disp(BERtheory)

semilogy(SNR,BERtheory,'b-',SNR,BER,'r*');
legend('Theoretical BER','Empirical BER');
xlabel('SNR (dB)');
ylabel('BER');
%title('Binary DPSK over Rayleigh Fading Channel');
title('Binary QPSK over Rayleigh Fading Channel');

qpskMod = comm.QPSKModulator;
rayChan = comm.RayleighChannel(
    SampleRate=bitRate,
    MaximumDopplerShift=4,
    PathDelays=[0 2e-5],
    AveragePathGains=[0 -9]);
cd = comm.ConstellationDiagram;