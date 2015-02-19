classdef FunctRegNode < Funct
    
    properties
        nx
        ix
        beta
        p
    end
    
    methods
        function obj = FunctRegNode(nx,ix,beta,p)
            obj.nx = nx;
            obj.ix = ix;
            obj.beta = beta;
            obj.p = p;
        end
        function [WI, XI, cost] = f(obj, d, WI, XI, cost)
            X = XI{1}.read(obj.ix, obj.nx);
            n = size(X,2);
            pi = sum(X,2) ./ size(X,2);
            if d == 0
                Y0 = obj.p.*log(obj.p./pi) + (1-obj.p).*log((1-obj.p)./(1-pi));
                cost = cost + obj.beta * sum(Y0);
            elseif d == 1
                Y1 = -obj.p ./ pi + (1 - obj.p) ./ (1 - pi);
                XI{d+1}.write(repmat(obj.beta * Y1 / n,1,n), obj.ix);
            elseif d == 2
                Y2 = obj.p ./ pi.^2 + (1 - obj.p) ./ (1 - pi).^2;
                XI{d+1}.write(repmat(obj.beta * Y2 / n,1,n), obj.ix);
            end
            assert(d>=0, 'Error: RegNode derivative d<0');
            assert(d<=2, 'Error: RegNode derivative d>2');
        end
    end
    
end