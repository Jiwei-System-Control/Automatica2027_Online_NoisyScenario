% This script simulates a continuous-time linear system under piecewise-constant
% inputs and estimates its system matrices A and B from the simulated data.
% The user specifies the process-noise level wl; set wl = 0 for a noise-free run.
% Two identification approaches are compared:
%   1. Filtered identification using lower-pass filtered states and inputs.
%   2. Unfiltered identification using sampled states, inputs, and backward
%      finite-difference estimates of the state derivative.
% The script displays the true matrices, both estimated matrix pairs, and the
% 2-norm estimation errors.

clear; clc; close all;

%% 1. Simulation settings
dt = 1e-8;                % Simulation time step [s].
T_loop = 0.1;             % Duration of each constant-input segment [s].
maxIterations = 6;        % Number of input segments to simulate.
current_time = 0;         % Current simulation time [s].
t_estimate = 1e-3;        % Window used for the backward finite-difference estimate [s].


%% 2. Continuous-time system and exact zero-order-hold discretization
% Continuous-time model:
%   dx/dt = A*x + B*u,
% where x is the state vector and u is the control-input vector.
A = [
    -0.0190   0.0825  -0.1005  -0.3206;
    -0.2154  -2.7859   1.2031  -0.0271;
     3.2527 -30.7871  -3.5418   0;
     0        0        1        0
];

B = [
     0.0065   0.0534;
    -0.6103   0.0020;
   -74.6355   0.5431;
     0        0
];

% Number of states and control inputs.
n = size(A, 1);
m = size(B, 2);

% Exact discretization under a zero-order hold over one time step:
%   x(k+1) = Ad*x(k) + Bd*u(k),
% where u is held constant over the interval of length dt.
Ad = expm(A * dt);
Bd = A \ (expm(A * dt) - eye(n)) * B;


%% 3. Lower-pass filter
% The state and input filters are defined by
%   dx_f/dt = -rho*x_f + x,
%   du_f/dt = -rho*u_f + u.
% Af and Bf are the exact discrete-time coefficients when the filter input
% is assumed constant over one simulation time step.
rho = 1;
Af = expm(-rho * dt);
Bf = (1 - expm(-rho * dt)) / rho;


%% 4. Initial conditions, noise level, and input sequence
x0 = [2; -1; 1; 0.5];   % Initial state of the system.
x = x0;

% Initial conditions for the filter states.
x_f2 = zeros(n, 1);
u = zeros(m, 1);
u_f2 = zeros(m, 1);

% Ask the user to specify the process-noise amplitude.
% Set wl = 0 to run a noise-free simulation. The effective noise increment
% also scales with sqrt(dt) and is injected through the input matrix B.
wl = input('Enter noise level wl (>= 0, e.g., 1e-5; enter 0 for no noise): ');
if ~isscalar(wl) || ~isreal(wl) || isnan(wl) || isinf(wl) || wl < 0
    error('The noise level wl must be a finite, nonnegative real scalar.');
end
dw = sqrt(dt);
fprintf('Using noise level wl = %.6g.\n', wl);

% Each column is the constant input applied during one simulation segment.
inputdata = [ 0.5, -0.2, 0,  0, 0.2, -0.5;
             -1.0,  0.0, 0.5, -0.5, 0.0,  1.0];
if size(inputdata, 2) < maxIterations
    error('inputdata must contain at least maxIterations columns.');
end


%% 5. Prepare data records and a memory-efficient state buffer
% Segment-end data are recorded once after each constant-input segment.
timeArray = [];       % Segment-end time stamps [s].
x_sArray = x0';       % System states sampled at segment ends (initial state included).
u_sArray = [];        % Inputs sampled at segment ends.
x_dArray = [];        % State-derivative estimates from backward finite differences.
x_f2Array = [];       % State-filter outputs sampled at segment ends.
u_f2Array = [];       % Input-filter outputs sampled at segment ends.

% The derivative estimate uses x(t) and x(t-t_estimate). Store only the
% most recent t_estimate/dt + 1 states instead of allocating the full
% trajectory. This avoids a multi-gigabyte array for dt = 1e-8 s.
steps = round(T_loop / dt);
t_estimate_steps = round(t_estimate / dt);
if abs(steps * dt - T_loop) > 10 * eps(max(T_loop, 1))
    error('T_loop must be an integer multiple of dt.');
end
if abs(t_estimate_steps * dt - t_estimate) > 10 * eps(max(t_estimate, 1))
    error('t_estimate must be an integer multiple of dt.');
end
if t_estimate_steps < 1 || t_estimate_steps >= steps
    error('t_estimate must be positive and shorter than T_loop.');
end
bufferLength = t_estimate_steps + 1;
stateBuffer = zeros(bufferLength, n);
globalStep = 0;


%% 6. Simulate the system under piecewise-constant inputs
for iter = 1:maxIterations
    % Apply the next control-input vector. It remains constant for T_loop.
    u = inputdata(:, iter);

    for i = 1:steps
        % Generate the Gaussian process-noise increment. The two noise
        % channels have relative amplitudes 0.25 and 1, respectively.
        w = wl * dw * B * [0.25 * randn(); randn()];

        % Update the system using the exact zero-order-hold model for the
        % deterministic part, plus the process-noise increment.
        x = Ad * x + Bd * u + w;

        % Update the filters. The state filter uses the newly computed state
        % at this step; the input filter is driven by the held input u.
        x_f2 = Af * x_f2 + Bf * x;
        u_f2 = Af * u_f2 + Bf * u;

        % Advance the clock and store the current state in a circular buffer.
        current_time = current_time + dt;
        globalStep = globalStep + 1;
        bufferIndex = mod(globalStep - 1, bufferLength) + 1;
        stateBuffer(bufferIndex, :) = x';
    end

    % Estimate the state derivative at the segment end using a backward
    % finite difference over t_estimate seconds (1 ms by default).
    currentIndex = mod(globalStep - 1, bufferLength) + 1;
    previousStep = globalStep - t_estimate_steps;
    previousIndex = mod(previousStep - 1, bufferLength) + 1;
    xd = (stateBuffer(currentIndex, :) - stateBuffer(previousIndex, :)) / t_estimate;

    % Record the state, input, filter outputs, and time at this segment end.
    timeArray = [timeArray; current_time];
    x_dArray = [x_dArray; xd];
    x_sArray = [x_sArray; x'];
    u_sArray = [u_sArray; u'];
    x_f2Array = [x_f2Array; x_f2'];
    u_f2Array = [u_f2Array; u_f2'];

    fprintf('Completed input segment %d at t = %.2f s.\n', iter, current_time);
end


%% 7. Estimate [A, B] using filtered data
% For a zero-initialized lower-pass filter driven by x(t), the filtered
% derivative satisfies
%   x_df(t) = x(t) - exp(-rho*t)*x(0) - rho*x_f(t).
% These correction terms retain the original source-code values for the six
% segment-end times 0.1, 0.2, ..., 0.6 s. If T_loop or maxIterations changes,
% update this vector consistently with the new segment-end times.
x02Array = [exp(-0.1 * rho) * x0';
            exp(-0.2 * rho) * x0';
            exp(-0.3 * rho) * x0';
            exp(-0.4 * rho) * x0';
            exp(-0.5 * rho) * x0';
            exp(-0.6 * rho) * x0'];
x_df2Array = x_sArray(2:end, :) - x02Array - rho * x_f2Array;

% Form the regression based on filtered state and input data:
%   x_df = [A, B] * [x_f; u_f].
% Right matrix division estimates the combined matrix [A, B].
% With the current settings, there are six segment-end samples and six
% regressors; the regressor matrix must be nonsingular for a unique estimate.
AB = x_df2Array' / [x_f2Array'; u_f2Array'];
A_hat_filtered = AB(:, 1:n);
B_hat_filtered = AB(:, n+1:n+m);
filteredEstimationError = norm([A, B] - AB);


%% 8. Estimate [A, B] using unfiltered data for comparison
% This baseline uses finite-difference estimates of dx/dt and the sampled
% states and inputs, without applying the state/input filters.
ABn = x_dArray' / [x_sArray(2:end, :)'; u_sArray'];
A_hat_unfiltered = ABn(:, 1:n);
B_hat_unfiltered = ABn(:, n+1:n+m);
unfilteredEstimationError = norm([A, B] - ABn);


%% 9. Display the estimated system matrices and estimation errors
% The true matrices A and B are shown first, followed by the estimates from
% the filtered and unfiltered data for direct comparison.
fprintf('\n============================================================\n');
fprintf('Noise level: wl = %.6g\n', wl);
fprintf('True system matrices\n');
fprintf('A =\n');
disp(A);
fprintf('B =\n');
disp(B);

fprintf('Estimated matrices using FILTERED data\n');
fprintf('A_hat_filtered =\n');
disp(A_hat_filtered);
fprintf('B_hat_filtered =\n');
disp(B_hat_filtered);
fprintf('2-norm error for [A, B] (filtered data): %.6e\n', filteredEstimationError);

fprintf('\nEstimated matrices using UNFILTERED data\n');
fprintf('A_hat_unfiltered =\n');
disp(A_hat_unfiltered);
fprintf('B_hat_unfiltered =\n');
disp(B_hat_unfiltered);
fprintf('2-norm error for [A, B] (unfiltered data): %.6e\n', unfilteredEstimationError);
fprintf('============================================================\n');