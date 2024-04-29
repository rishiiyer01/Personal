
%unstructured mesh generation


% Cylinder parameters
radius = 0.5;
center = [0, 0];

theta = linspace(0, 2*pi, 50);
x_cylinder = radius * cos(theta) + center(1);
y_cylinder = radius * sin(theta) + center(2);
innercons=[x_cylinder',y_cylinder'];
% Domain parameters
domain_size = 10;
x_min = -domain_size;
x_max = 2*domain_size;
y_min = -domain_size;
y_max = domain_size;

n_outer = 100;
theta_outer = linspace(0, 2*pi, n_outer+1);
x_outer = 0.5*domain_size * cos(theta_outer);
y_outer = 0.5*domain_size * sin(theta_outer);

x_points = [x_cylinder, x_outer(1:end-1)];
y_points = [y_cylinder, y_outer(1:end-1)];

n_random = 1000;
x_random = (x_max - x_min) * rand(1, n_random) + x_min;
y_random = (y_max - y_min) * rand(1, n_random) + y_min;

distance = sqrt((x_random - center(1)).^2 + (y_random - center(2)).^2);
x_random = x_random(distance > radius);
y_random = y_random(distance > radius);

n_random2=1000;
x_random2=0.5*(2*x_max - x_min) * rand(1, n_random2) + 0.5*x_min;
y_random2=0.5*(y_max - y_min) * rand(1, n_random2) + 0.5*y_min;

distance2=sqrt((x_random2 - center(1)).^2 + (y_random2 - center(2)).^2);
x_random2= x_random2(distance2 > radius);
y_random2=y_random2(distance2 > radius);


xbottom_domain=linspace(x_min,x_max,50);
ybottom_domain=y_min*ones(size(xbottom_domain));

xtop_domain=linspace(x_min,x_max,50);
ytop_domain=y_max*ones(size(xtop_domain));

yleft_domain=linspace(y_min,y_max,50);
xleft_domain=x_min*ones(size(yleft_domain));

yright_domain=linspace(y_min,y_max,50);
xright_domain=x_max*ones(size(yright_domain));




x_all = [x_points, x_random,x_random2,xbottom_domain,xtop_domain,xleft_domain,xright_domain];
y_all = [y_points, y_random,y_random2,ybottom_domain,ytop_domain,yleft_domain,yright_domain];

DT = delaunayTriangulation(x_all', y_all');

triplot(DT);
axis equal;


triangles = DT;
numCells = size(triangles, 1);


vertices = [x_all(:), y_all(:)];

cells = cell(numCells, 1);


for i = 1:numCells
    % Get the indices of the vertices of the current triangle
    vertexIndices = triangles(i, :);
    numFaces = 3;
    % Get the coordinates of the vertices of the current triangle
    cellVertices = vertices(vertexIndices, :);
    
    % Compute the area of the current triangle
    cellArea = computeTriangleArea(cellVertices);
    
    % Compute the centroid of the current triangle
    cellCentroid = mean(cellVertices);
    
    % Store the cell information in the cell array
    cells{i}.vertices = cellVertices;
    cells{i}.area = cellArea;
    cells{i}.centroid = cellCentroid;
    for j = 1:numFaces
        vertex1 = cellVertices(j, :);
        vertex2 = cellVertices(mod(j, numFaces) + 1, :);
        midpt=0.5*vertex1+0.5*vertex2;
        faceNormalguess=midpt-cellCentroid;
        faceVector = vertex2 - vertex1;
        faceNormal = [faceVector(2), -faceVector(1)];
        if dot(faceNormalguess,faceNormal)>0
            faceNormal = faceNormal / norm(faceNormal);
        else
            faceNormal=-faceNormal/norm(faceNormal);
        end
        
        
        cells{i}.faceNormals(j, :) = faceNormal;
    end

end

%neighbors = neighbors(DT,:);
for i = 1:numCells
    cells{i}.neighbors = neighbors(DT,i);
end




for i = 1:numCells
    cells{i}.cellType = 0; % Initialize cell type to 0 (interior cell)
end

for i = 1:numCells
    cellVertices = cells{i}.vertices;
    verticesOnCylinder = sqrt((cellVertices(:,1) - center(1)).^2 + (cellVertices(:,2) - center(2)).^2) <= radius;
    if sum(verticesOnCylinder) >= 2
        cells{i}.cellType = 5; % Cylinder boundary cell
    end
end

for i = 1:numCells
    cellVertices = cells{i}.vertices;
    verticesOnLeftEdge = cellVertices(:,1) == x_min;
    verticesOnRightEdge = cellVertices(:,1) == x_max;
    verticesOnBottomEdge = cellVertices(:,2) == y_min;
    verticesOnTopEdge = cellVertices(:,2) == y_max;
    
    if sum(verticesOnLeftEdge) >= 2
        cells{i}.cellType = 1; % Left boundary cell
    elseif sum(verticesOnRightEdge) >= 2
        cells{i}.cellType = 2; % Right boundary cell
        i
    elseif sum(verticesOnBottomEdge) >= 2
        cells{i}.cellType = 3; % Bottom boundary cell
    elseif sum(verticesOnTopEdge) >= 2
        cells{i}.cellType = 4; % Top boundary cell
    end
end

for i = 1:numCells
    cellVertices = cells{i}.vertices;
    verticesOnBottomLeftCorner = (cellVertices(:,1) == x_min) & (cellVertices(:,2) == y_min);
    verticesOnBottomRightCorner = (cellVertices(:,1) == x_max) & (cellVertices(:,2) == y_min);
    verticesOnTopLeftCorner = (cellVertices(:,1) == x_min) & (cellVertices(:,2) == y_max);
    verticesOnTopRightCorner = (cellVertices(:,1) == x_max) & (cellVertices(:,2) == y_max);
    
    if sum(verticesOnBottomLeftCorner) >= 2
        cells{i}.cellType = 6; % Bottom-left corner cell
    elseif sum(verticesOnBottomRightCorner) >= 2
        cells{i}.cellType = 7; % Bottom-right corner cell
    elseif sum(verticesOnTopLeftCorner) >= 2
        cells{i}.cellType = 8; % Top-left corner cell
    elseif sum(verticesOnTopRightCorner) >= 2
        cells{i}.cellType = 9; % Top-right corner cell
    end
end

function area = computeTriangleArea(vertices)
    v1 = vertices(2, :) - vertices(1, :);
    v2 = vertices(3, :) - vertices(1, :);
    area = 0.5 * abs(v1(1) * v2(2) - v1(2) * v2(1));
end