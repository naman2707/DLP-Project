function [fI,f0,nw,nx] = factoryLayeredNet(H,func,funclast,funcerror,lambda,l,beta,p)
    if isempty(func)
        func = Sigmoid;
    end
    if isempty(funclast)
        funclast = Sigmoid;
    end
    if isempty(funcerror)
        funcerror = @FunctSquareError;
    end
    if isempty(l)
        l = 2;
    end
    fI = cell(0);
    ix = 0;
    iw = 0;
    ib = 0;
    for i = 2:length(H)
        ib = ib + H(i-1) * H(i);
    end
    for i = 2:length(H)
        m = H(i-1);
        n = H(i);
        fI{end+1} = FunctTranspose(m,n,iw,ib,ix,ix+m);
        if lambda ~= 0
            fI{end+1} = FunctRegWeight(m*n,iw,lambda,l);
        end
        if i ~= length(H)
            fI{end+1} = FunctUnit(n,n,ix+m,ix+m+n,func);
            if beta ~= 0
                fI{end+1} = FunctRegNode(n,ix+m+n,beta,p);
            end
        else
            fI{end+1} = FunctUnit(n,n,ix+m,ix+m+n,funclast);
        end
        iw = iw + m*n;
        ib = ib + n;
        ix = ix + m + n;
    end
    f0 = funcerror(n,ix);
    nw = ib;
    nx = ix+n;
    
%     for i = 1:length(fI)-1
%         assert(fI{i}.iy == fI{i+1}.ix);
%         assert(fI{i}.ny == fI{i+1}.nx);
%     end
%     assert(fI{end}.iy == f0.ix);
%     assert(fI{end}.ny == f0.nx);
end

