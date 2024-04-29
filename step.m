clc;
clear;
close all;
triangle_mesh_cyl


% Initialize velocity and pressure fields
u = zeros(numCells, 1);
v = zeros(numCells, 1);
p = zeros(numCells, 1);

% Set convergence criteria
maxIterations = 1000;
tolerance = 1e-6;

% Set time step and final time
dt = 0.01;
finalTime = 1.0;

% Initialize storage for time-dependent solutions
u_time = zeros(numCells, length(0:dt:finalTime));
v_time = zeros(numCells, length(0:dt:finalTime));
p_time = zeros(numCells, length(0:dt:finalTime));
t=0:dt:finalTime;
% Time loop
for n = 1:length(0:dt:finalTime)-1
    % Set the initial guess as the previous time step solution
    u_guess = u_time(:, n);
    v_guess = v_time(:, n);
    p_guess = p_time(:, n);
    
    % Reset the iteration count and convergence criteria
    iter = 1;
    u_change = zeros(length(t)-1);
    
    % Begin iterative SIMPLE algorithm
    while iter <= maxIterations && abs(u_change(end)) > tolerance
        u_guess_old = u_guess;
        v_guess_old = v_guess;
        p_guess_old = p_guess;
        
        % Solve momentum equations
        [u_star, v_star] = solveMomentumEquations(u_guess, v_guess, p_guess, cells, dt);
        
        % Solve pressure correction equation
        p_prime = solvePressureCorrectionEquation(u_star, v_star, cells, dt);
        
        % Update pressure
        p_guess = p_guess + p_prime;
        
        % Correct velocity field
        [u_guess, v_guess] = correctVelocityField(u_star, v_star, p_prime, cells, dt);
        
        % Calculate changes in u, v, and p
        u_change(iter) = sqrt(sum((u_guess - u_guess_old).^2)) / numCells;
        % i didnt want to compute residuals for all variables
        
        % Increment iteration count
        iter = iter + 1;
    end
    
    % Store the solution at the current time step
    u_time(:, n+1) = u_guess;
    v_time(:, n+1) = v_guess;
    p_time(:, n+1) = p_guess;
end

% Function to solve momentum equations
function [u_star, v_star] = solveMomentumEquations(u, v, pstar, cells, dt)

    
    %first u equation
    
    for j=length(cells)
        %reorder neighbors
        normal=zeros(3,2);
        for n=1:3
            neighbor_id=cells{j}.neighbors(n);
            centervec=cells{neighbor_id}.centroid-cells{j}.centroid;
            if dot(cells{j}.faceNormals(1,:),centervec)>0
                normal(n,:)=cells{j}.faceNormals(1,:);
            elseif dot(cells{j}.faceNormals(2,:),centervec)>0
                normal(n,:)=cells{j}.faceNormals(2,:);
            elseif dot(cells{j}.faceNormals(3,:),centervec)>0
                normal(n,:)=cells{j}.faceNormals(3,:);
            else
                print="error"
            end
        end
        if cells{j}.cellType==1
            %left boundary
            for neighbors=cells{j}.neighbors
                if isnan(neighbors)
                else

                end
            end

        end
    end
        
        
end


% Function to solve pressure correction equation
function p_prime = solvePressureCorrectionEquation(u_star, v_star, cells, dt)
    % Implement the discretization and solution of pressure correction equation
    % based on your specific problem and boundary conditions
    % Include temporal discretization terms (e.g., backward Euler)
    % Return the pressure correction field (p_prime)
end

% Function to correct velocity field
function [u, v] = correctVelocityField(u_star, v_star, p_prime, cells, dt)
    % Implement the velocity correction step based on the pressure correction
    % Return the corrected velocity field (u, v)
end