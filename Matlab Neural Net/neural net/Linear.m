classdef Linear < SimpleFunct
    
    properties
    end
    
    methods
        function Y = f(obj, X)
            Y = X;
        end
        function dX = df(obj, d, X, dY, c)
            dX = zeros(size(dY));
            if d == 1
                dX = dY;
            end
        end
    end
    
end