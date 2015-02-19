classdef Funct
    
    properties
    end
    
    methods
        function X = readShape(obj, XI, i, r, c)
            X = reshape(XI.read(i,r*c), r, c);
        end
    end
    
    methods (Abstract)
        [WI, XI, cost] = f(obj, d, WI, XI, cost)
    end
    
end