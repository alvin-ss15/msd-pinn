function [t, x, xdot, y, ydot] = simulate_msd(sim_time, Ts, u1, dist, u3, m, k, kp, d, x0, xdot0)
% Simplified mass-spring-damper simulation for MPC

% Create time vector
t = 0:Ts:sim_time;
N = length(t);

% Handle different input types
inputs = {u1, dist, u3};
for i = 1:3
    if isa(inputs{i}, 'function_handle')
        inputs{i} = arrayfun(inputs{i}, t);
    elseif isscalar(inputs{i})
        inputs{i} = inputs{i} * ones(1, N);
    end
end
[u1, dist, u3] = deal(inputs{:});

% Initialize state arrays
x = zeros(N, 3);
xdot = zeros(N, 3);
y = zeros(N, 3); 
ydot = zeros(N, 3);

% Set initial conditions
x(1, :) = x0;
xdot(1, :) = xdot0;
y(1, :) = x0;
ydot(1, :) = xdot0;

% Main simulation loop
for k = 1:N-1
    % Integrate with ZOH inputs
    [~, states] = ode45(@(t, state) dynamics(state, u1(k), dist(k), u3(k), m, k, kp, d), ...
                        [0 Ts], [x(k, :), xdot(k, :)]);
    
    x(k+1, :) = states(end, 1:3);
    xdot(k+1, :) = states(end, 4:6);
    y(k+1, :) = x(k, :);      % All 3 positions with delay
    ydot(k+1, :) = xdot(k, :); % All 3 velocities with delay
end
end

function dstate = dynamics(state, u1, dist, u3, m, k, kp, d)
% System dynamics for 3-mass spring-damper
x = state(1:3);
xdot = state(4:6);

% Calculate forces and accelerations
x1ddot = (1/m) * (k*(-2*x(1) + x(2)) + kp*(-x(1)^3 + (x(2)-x(1))^3) + d*(xdot(2) - 2*xdot(1)) + u1);
x2ddot = (1/m) * (k*(x(1) - 2*x(2) + x(3)) + kp*((x(3)-x(2))^3 - (x(2)-x(1))^3) + d*(xdot(1) - 2*xdot(2)) + dist);
x3ddot = (1/m) * (k*(x(2) - x(3)) + kp*(x(2)-x(3))^3 + d*(xdot(2) - xdot(3)) + u3);

dstate = [xdot; x1ddot; x2ddot; x3ddot];
end