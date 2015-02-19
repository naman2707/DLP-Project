classdef FunctUnit < Funct
    
    properties
        nx
        ny
        ix
        iy
        func
    end
    
    methods
        function obj = FunctUnit(nx,ny,ix,iy,func)
            obj.nx = nx;
            obj.ny = ny;
            obj.ix = ix;
            obj.iy = iy;
            obj.func = func;
        end
        function [WI, XI, cost] = f(obj, d, WI, XI, cost)
            X = XI{1}.read(obj.ix, obj.nx);
%             assert(sum(X(:))~=0,'Error: FunctUnit X=0');
            if d == 0
                XI{1}.write(obj.func.f(X), obj.iy);
            else
                dY = XI{2}.read(obj.iy, obj.ny);
                if d == 1
                    dYdX = obj.func.df(1,X,dY); 
                    XI{d+1}.write(dYdX, obj.ix);
                else
                    dY = XI{2}.read(obj.iy, obj.ny);
                    ddY = XI{d+1}.read(obj.iy, obj.ny);
                    dYddX = obj.func.df(2,X,dY);
                    ddYdX = obj.func.df(1,X,ddY,2);
                    XI{d+1}.write(dYddX + ddYdX, obj.ix);
%                 for i = 1:d
%                     dY = XI{d+2-i}.read(obj.iy, obj.ny);
%                     dX = obj.func.df(i,X,dY);
%                     a = factorial(d-1)/(factorial(i-1)*factorial(d-i));
%                     XI{d+1}.write(a * dX, obj.ix);
%                 end
                end
            end
            assert(d<=2,'Error: FunctUnit d>2');
            assert(d<=2,'Error: FunctUnit d<0');
        end
    end
    
end