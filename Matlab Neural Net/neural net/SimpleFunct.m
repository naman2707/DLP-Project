classdef SimpleFunct
    
    properties
    end
    
    methods (Abstract)
        Y = f(X)
        dX = df(i,X,dY,c)
    end
    
end