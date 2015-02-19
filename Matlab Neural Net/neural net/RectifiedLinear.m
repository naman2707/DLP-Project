classdef RectifiedLinear < SimpleFunct
    
    properties
    end
    
    methods
        function Y = f(obj, X)
            Y = max(X,zeros(size(X)));
        end
        function dX = df(obj, d, X, dY, c)
            if nargin < 5
                c = 1;
            end
            if d == 1
                dX = X>0;
            else
                dX = zeros(size(dY));
            end
            dX = dX.^c .* dY;
            assert(d>0, 'Error: RectifiedLinear derivative d<=0');
        end
    end
    
end