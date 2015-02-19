function dX = computeNumericalGradient(d, f, X)
dX = zeros(size(X));
EPSILON = 10.^(-digits/(2*(2+d)));
for i = 1:length(dX)
    if mod(i,100) == 0
        i
    end
    r = zeros(size(X));
    r(i) = EPSILON;
    dXi = zeros(d+1,1);
    for j = 0:d
        dXi(j+1) = f(X + r * (-d + 2*j));
    end
    for j = d:-1:1
        dXi(1:j) = (dXi(2:j+1) - dXi(1:j)) / (2 * EPSILON);
    end
    dX(i) = dXi(1);
end
end
