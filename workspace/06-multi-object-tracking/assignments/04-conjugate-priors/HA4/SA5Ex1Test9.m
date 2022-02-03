clear all; close all; clc;

%Choose object detection probability
P_D = 0.9;
%Choose clutter rate
lambda_c = 10;
%Choose object survival probability
P_S = 0.99;
%Create sensor model
%Range/bearing measurement range
range_c = [-1000 1000;-pi pi];
sensor_model = modelgen.sensormodel(P_D,lambda_c,range_c);
        
%Create nonlinear motion model (coordinate turn)
T = 1;
sigmaV = 1;
sigmaOmega = pi/180;
motion_model = motionmodel.ctmodel(T,sigmaV,sigmaOmega);
        
%Create nonlinear measurement model (range/bearing)
sigma_r = 5;
sigma_b = pi/180;
s = [300;400];
meas_model = measmodel.rangebearingmeasmodel(sigma_r, sigma_b, s);
        
%Creat ground truth model
nbirths = 4;
K = 10;
tbirth = zeros(nbirths,1);
tdeath = zeros(nbirths,1);
        
initial_state(1).x = [0; 0; 5; 0; pi/180];       tbirth(1) = 1;   tdeath(1) = 50;
initial_state(2).x = [20; 20; -20; 0; pi/90];    tbirth(2) = 20;  tdeath(2) = 70;
initial_state(3).x = [-20; 10; -10; 0; pi/360];  tbirth(3) = 40;  tdeath(3) = 90;
initial_state(4).x = [-10; -10; 8; 0; pi/270];   tbirth(4) = 60;  tdeath(4) = K;

birth_model = repmat(struct('w',log(0.03),'x',[],'P',diag([1 1 1 1*pi/90 1*pi/90].^2)),[1,4]);
birth_model(1).x = [0; 0; 5; 0; pi/180];
birth_model(2).x = [20; 20; -20; 0; pi/90];
birth_model(3).x = [-20; 10; -10; 0; pi/360];
birth_model(4).x = [-10; -10; 8; 0; pi/270];

%Generate true object data (noisy or noiseless) and measurement data
ground_truth = modelgen.groundtruth(nbirths,[initial_state.x],tbirth,tdeath,K);
ifnoisy = 0;
objectdata = objectdatagen(ground_truth,motion_model,ifnoisy);
measdata = measdatagen(objectdata,sensor_model,meas_model);

%Object tracker parameter setting
P_G = 0.999;            %gating size in percentage
w_min = 1e-3;           %hypothesis pruning threshold
merging_threshold = 2;  %hypothesis merging threshold
M = 100;                %maximum number of hypotheses kept
r_min = 1e-3;           %Bernoulli component pruning threshold
r_recycle = 0.1;        %Bernoulli component recycling threshold
r_estimate = 0.4;       %Threshold used to extract estimates from Bernoullis
density_class_handle = feval(@GaussianDensity);    %density class handle
tracker = multiobjectracker();
tracker = tracker.initialize(density_class_handle,P_S,P_G,meas_model.d,w_min,merging_threshold,M,r_min,r_recycle,r_estimate);

%Create a class instance
PMBM_ref = PMBMfilter();
PMBM_ref = PMBM_ref.initialize(tracker.density,birth_model);


