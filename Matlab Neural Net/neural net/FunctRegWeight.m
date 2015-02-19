classdef FunctRegWeight < Funct
    
    properties
        nw
        iw
        lambda
        l
    end
    
    methods
        function obj = FunctRegWeight(nw,iw,lambda,l)
            obj.nw = nw;
            obj.iw = iw;
            obj.lambda = lambda;
            obj.l = l;
        end
        function [WI, XI, cost] = f(obj, d, WI, XI, cost)
            W = WI{1}.read(obj.iw, obj.nw);
            if d == 0
                cost = cost + obj.lambda * sum(abs(W).^obj.l);
            else
                p = prod(obj.l+1-(1:d));
                if p ~= 0
                    WI{d+1}.write(obj.lambda * p * sign(W).*abs(W).^(obj.l-d), obj.iw);
                end
            end
        end
    end
    
end