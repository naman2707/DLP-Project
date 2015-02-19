classdef DataHandle < handle
    
    properties
        XI
    end
    
    methods
        function obj = DataHandle(XI)
            obj.XI = XI;
        end
        function X = read(obj, i, w)
            X = obj.XI(i+(1:w),:);
        end
        function write(obj, X, i)
            iter = i+(1:size(X,1));
            obj.XI(iter,:) = obj.XI(iter,:) + X;
        end
    end
    
end