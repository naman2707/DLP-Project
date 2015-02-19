classdef FunctTranspose < Funct
    
    properties
        nx
        ny
        iw
        ib
        ix
        iy
    end
    
    methods
        function obj = FunctTranspose(nx,ny,iw,ib,ix,iy)
            obj.nx = nx;
            obj.ny = ny;
            obj.iw = iw;
            obj.ib = ib;
            obj.ix = ix;
            obj.iy = iy;
        end
        function [WI, XI, cost] = f(obj, d, WI, XI, cost)
            W = obj.readShape(WI{1}, obj.iw, obj.ny, obj.nx);
            b = obj.readShape(WI{1}, obj.ib, obj.ny, 1);
            X = XI{1}.read(obj.ix, obj.nx);
            if d == 0
                XI{1}.write(W * X + repmat(b,1,size(X,2)), obj.iy);
            elseif d >= 1
                dY = XI{d+1}.read(obj.iy, obj.ny);
                dW = dY * X'.^d;
                WI{d+1}.write(dW(:), obj.iw);
                WI{d+1}.write(sum(dY,2), obj.ib);
                XI{d+1}.write(W'.^d * dY, obj.ix);
            end
        end
    end
    
end