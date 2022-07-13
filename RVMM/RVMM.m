function RVMM(Global)
% <algorithm> <R>
% Surrogate-assisted RVEA
% alpha ---  2 --- The parameter controlling the rate of change of penalty
% wmax  --- 20 --- Number of generations before updating Kriging models
% mu    ---  1 --- Number of re-evaluated solutions at each generation

%------------------------------- Reference --------------------------------
% T. Chugh, Y. Jin, K. Miettinen, J. Hakanen, and K. Sindhya, A surrogate-
% assisted reference vector guided evolutionary algorithm for
% computationally expensive many-objective optimization, IEEE Transactions
% on Evolutionary Computation, 2018, 22(1): 129-142.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Cheng He

%% Parameter setting
rng('shuffle');
rng(randperm(2^32-1,1),'twister');


[alpha,wmax,mu] = Global.ParameterSet(2,100,1);

%% Generate the reference points and population
[V0,Global.N] = UniformPoint(Global.N,Global.M);
V     = V0;
Vbb     = V0;
NI    = 11*Global.D-1;

P     = lhsamp(NI,Global.D);
% P  = lhsdesign(NI, Global.D,'criterion','maximin','iterations',1000) ;
A2    = INDIVIDUAL(repmat(Global.upper-Global.lower,NI,1).*P+repmat(Global.lower,NI,1));

A1    = A2;
THETA = 5.*ones(Global.M,Global.D);
Model = cell(1,Global.M);
NumV = 5;
gen = 1;
flag_dace = 1;

kk = 0.5;
%% Optimization
while Global.NotTermination(A2)
    % Refresh the model and generate promising solutions
    A1Dec = A1.decs;
    A1Obj = A1.objs;
    [c,ia,ic] = unique(A1Obj,'rows','stable');
    A1Obj = A1Obj(ia,:);
    A1Dec = A1Dec(ia,:);
    A1 = A1(ia);
	
		A1(find(sum(isnan(A1Obj),2)>0))=[];
	A1Dec(find(sum(isnan(A1Obj),2)>0),:)=[];
	A1Obj(find(sum(isnan(A1Obj),2)>0),:)=[];
	
	A1(find(sum(isinf(A1Obj),2)>0))=[];
	A1Dec(find(sum(isinf(A1Obj),2)>0),:)=[];
	A1Obj(find(sum(isinf(A1Obj),2)>0),:)=[];
    
    [c,ia,ic] = unique(A1Obj,'rows');
    A1Obj = A1Obj(ia,:);
    A1Dec = A1Dec(ia,:);
    A1 = A1(ia);
    [c,ib,ic] = unique(A1Dec,'rows');
    A1Obj = A1Obj(ib,:);
    A1Dec = A1Dec(ib,:);
    A1 = A1(ib);
	
    
    V0 = Vbb;
    
    
    for i = 1 : Global.M
        
        
        % The parameter 'regpoly1' refers to one-order polynomial
        % function, and 'regpoly0' refers to constant function. The
        % former function has better fitting performance but lower
        % efficiency than the latter one
        
        %         try
        
        if 1==flag_dace
            dmodel     = dacefit(A1Dec,A1Obj(:,i),'regpoly0','corrgauss',THETA(i,:),1e-5.*ones(1,Global.D),100.*ones(1,Global.D));
            Model{i}   = dmodel;
            THETA(i,:) = dmodel.theta;
        else
            lower_bound = Global.lower;
            upper_bound = Global.upper;
            
            if gen == 1
                % initial learning
                num_vari = Global.D;
                kriging_model{i} = kriging_theta_train(A1Dec,A1Obj(:,i),lower_bound,upper_bound,1*ones(1,num_vari),0.000001*ones(1,num_vari),100*ones(1,num_vari));
                
            else
                if ~isempty(New)
                    infill_x = New.decs;
                    infill_y = New.objs;
                    % incremental learning
                    kriging_model{i} = kriging_incremental(kriging_model{i},infill_x,infill_y(:,i),A1Dec(1:end-1,:),A1Obj(1:end-1,i),lower_bound,upper_bound);
                end
            end
        end
        %         catch exception
        %             A1Obj(:,i)
        %         end
    end
    gen = 0;
    PopDec1 = A1Dec;
    w      = 1;
    
    A2Obj = A2.objs;
    A2Dec = A2.decs;
    [c,ia,ic] = unique(A2Obj,'rows','stable');
    A2Obj = A2Obj(ia,:);
    %             A2Dec = A2Dec(ia,:);
    %     A2 = A2(ia);
    
    A2Obj = A2Obj(NDSort(A2Obj,1)==1,:);
    zmin = min(A2Obj,[],1);
    A2Obj_temp = A2Obj - repmat(min(zmin,[],1),size(A2Obj,1),1);
    if size(A2Obj,1)>=2
        scale = (max(A2Obj,[],1)-min(A2Obj,[],1));
        V_1 = V0.*(max(A2Obj,[],1)-min(A2Obj,[],1));
    else
         scale = ones(1,Global.M);
        V_1 = V0;
    end
    
    Angle   = acos(1-pdist2(A2Obj_temp,V_1,'cosine'));
    [~,associate] = min(Angle,[],2);
    active  = unique(associate,'stable');
    Va        = V_1(active,:);
    NCluster = min(5,size(Va,1));
    [IDX,C]   = kmeans(V0(active,:),NCluster);
    V1 = [];
    ids = [];
    for i=1:NCluster
        EC = find(IDX==i);
        id = EC(randperm(size(EC,1),1),1);
        V1(i,:) = Va(id,:);
        ids = [ids;id];
    end
    
    if size(V1,1)<5
        notS = setdiff(1:size(V_1,1),active(ids));
        V1 = [V1;V_1( notS(randperm(size(notS,2),5-size(V1,1))),:)];
    end
    
        
    WPopDec = [];
    WPopObj = [];
    WMSE = [];
    while w <= wmax
        drawnow();
        MatingPool = randi(size(PopDec1,1),1,Global.N);
        OffDec  = GA(PopDec1(MatingPool,:));
        
        pop_candi = [];
        NP = size(OffDec,1);
        for ii = 1 : NP
            if min(sqrt(sum((OffDec(ii,:) - [A1Dec;pop_candi]).^2,2)))>1E-6
                pop_candi = [pop_candi;OffDec(ii,:)];
            end
        end
        OffDec = pop_candi;
        
        %             OffDec = GA(PopDec);
        PopDec1 = [PopDec1;OffDec];
        [N,~]  = size(PopDec1);
        PopObj1 = zeros(N,Global.M);
        MSE1    = zeros(N,Global.M);
        if 1==flag_dace
            for i = 1: N
                for j = 1 : Global.M
                    [PopObj1(i,j),~,MSE1(i,j)] = predictor(PopDec1(i,:),Model{j});
                end
            end
            
        else
            %             for i = 1: N
            for j = 1 : Global.M
                [PopObj1(:,j), MSE1(:,j) ]  = kriging_predictor(PopDec1,kriging_model{j},A1Dec,A1Obj(:,j),Global.lower,Global.upper);
                %                 end
            end
        end
        
        MSE1 = max(MSE1,0);
        S_ = sqrt(MSE1);
        MSE1 = S_.*(MSE1<=1)+MSE1.*(MSE1>1);
		
        %MSE1 = S_;
        PopObj1_b = PopObj1;
        MSE1_b = MSE1;
        PopObj1 = PopObj1+kk*MSE1;
        
        zmin = min([zmin;PopObj1],[],1);
 
        [index]  = KEnvironmentalSelection(PopObj1,[V1;[]],(w/wmax)^alpha);
        
        PopDec1 = PopDec1(index,:);
        PopObj1 = PopObj1(index,:);
        %            plot(PopObj1(:,1),PopObj1(:,2),'bs')
        MSE1 = MSE1(index,:);
        
        [~,ib]= intersect(PopDec1,A2.decs,'rows');
        PopDec1_ = PopDec1;
        PopDec1_(ib,:)=[];
        %         str = '*******************full******************'
        if isempty(PopDec1_)
            offobj1 = PopObj1_b(size(PopObj1_b,1)-size(OffDec,1)+1:end,:);
            offmse1 = MSE1_b(size(PopObj1_b,1)-size(OffDec,1)+1:end,:);
            [frontNo,~] = NDSort(offobj1,size(offobj1,1));
            solId = find(frontNo==1);
            PopDec1 =[PopDec1; OffDec(solId,:)];
            PopObj1 =[PopObj1; offobj1(solId,:)+kk*offmse1(solId,:)];
            MSE1 =[MSE1; offmse1(solId,:)];
            %             str = '---------------empty------------------'
        end
        
        
        % Adapt referece vectors
        if ~mod(w,ceil(wmax*0.1))&& size(unique(PopObj1,'rows'),1)>2
            V1 = V1./scale;
            scale = max(PopObj1,[],1)-min(PopObj1,[],1);
            V1 = V1.*scale;
        end
        w = w + 1;
        
    end
    
    PopObj1 = PopObj1-kk*MSE1;
    
    [c,ia,ic] = unique(PopObj1,'rows','stable');
    if ~isempty(ia)
        PopObj1 = PopObj1(ia,:);
        PopDec1 = PopDec1(ia,:);
        MSE1 = MSE1(ia,:);
    end
    
    
    [~,ib]= intersect(PopDec1,A2.decs,'rows');
    PopObj1(ib,:)=[];
    PopDec1(ib,:)=[];
    MSE1(ib,:)=[];
    

    w=1;
    PopDec = A1Dec;

    
    
    while w <= wmax
        drawnow();
       
        
        if size(PopDec,1)<Global.N
            MatingPool = randi(size(PopDec,1),1,Global.N);
            OffDec  = GA(PopDec(MatingPool,:));
        else
            OffDec = GA(PopDec);
        end
        
        pop_candi = [];
        NP = size(OffDec,1);
        for ii = 1 : NP
            if min(sqrt(sum((OffDec(ii,:) - [A1Dec;pop_candi]).^2,2)))>1E-6
                pop_candi = [pop_candi;OffDec(ii,:)];
            end
        end
        OffDec = pop_candi;
        
        
        
        PopDec = [PopDec;OffDec];
        [N,~]  = size(PopDec);
        PopObj = zeros(N,Global.M);
        MSE    = zeros(N,Global.M);
        
        if 1==flag_dace
            for i = 1: N
                for j = 1 : Global.M
                    [PopObj(i,j),~,MSE(i,j)] = predictor(PopDec(i,:),Model{j});
                end
            end
        else
            %             for i = 1: N
            for j = 1 : Global.M
                [PopObj(:,j),MSE(:,j)]  = kriging_predictor(PopDec,kriging_model{j},A1Dec,A1Obj(:,j),Global.lower,Global.upper);
                %                 end
            end
        end
        MSE = max(MSE,0);
        S_ = sqrt(MSE);
        
        MSE = S_.*(MSE<=1)+MSE.*(MSE>1);
		
		%MSE = S_;
        PopObj_b = PopObj;
        PopDec_b = PopDec;
        MSE_b = MSE;
        
        PopObj = PopObj+kk*MSE;
        
        if w==1
            PopObj_ = PopObj(NDSort(PopObj,1)==1,:);
            scale = (max(PopObj_,[],1)-min(PopObj_,[],1));
            zeroid= find(scale==0);
            scale(:,zeroid) =10^(-6);
            V = V0.*scale;
            
            Angle   = acos(1-pdist2(PopObj_, V,'cosine'));
            [~,associate] = min(Angle,[],2);
            active  = unique(associate,'stable');
            Va        =  V(active,:);
            if size(Va,1)<Global.N
                PopObj_temp = (PopObj_-min(PopObj_,[],1))./scale;
                Vadd = PopObj_temp(randperm(size(PopObj_temp,1),min(Global.N-size(Va,1),size(PopObj_temp,1))),:);
                V0 = [V0;Vadd];
                 V =  V0.*scale;
            end
        end
        
        
        index  = KEnvironmentalSelection(PopObj,V,(w/wmax)^alpha);
        
        PopDec = PopDec(index,:);
        PopObj = PopObj(index,:);
        MSE = MSE(index,:);
        
        [~,ib]= intersect(PopDec,A2.decs,'rows');
        PopDec_ = PopDec;
        PopDec_(ib,:)=[];
        if isempty(PopDec_)
            offobj = PopObj_b(size(PopObj_b,1)-size(OffDec,1)+1:end,:);
            offmse = MSE_b(size(PopObj_b,1)-size(OffDec,1)+1:end,:);
            [frontNo,~] = NDSort(offobj,size(offobj,1));
            solId = find(frontNo==1);
            PopDec =[PopDec; OffDec(solId,:)];
            PopObj =[PopObj;offobj(solId,:)+kk*offmse(solId,:)];
            MSE =[MSE; offmse(solId,:)];
        end
        
        % Adapt referece vectors
        if ~mod(w,ceil(wmax*0.1)) && size(unique(PopObj,'rows'),1)>2
            V = V0.*(max(PopObj,[],1)-min(PopObj,[],1));
        end
        %           plot( V(:,1), V(:,2),'bs')
        w = w + 1;
        WPopDec = [WPopDec;PopDec];
        WPopObj = [WPopObj;PopObj];
        WMSE = [WMSE;MSE];
        
        
    end
	 
    PopObj = WPopObj;
    PopDec = WPopDec;
    MSE = WMSE;
    
    PopObj = PopObj-kk*MSE;
    
    [c,ia,ic] = unique(PopObj,'rows','stable');
    if ~isempty(ia)
        PopObj = PopObj(ia,:);
        PopDec = PopDec(ia,:);
        MSE = MSE(ia,:);
    end
    
    [~,ib]= intersect(PopDec,A2.decs,'rows');
    
    PopObj(ib,:)=[];
    PopDec(ib,:)=[];
    MSE(ib,:)=[];
    
    
    NumVf =[];
    PopNew    = KrigingSelect(PopDec,PopObj,MSE,[V],V0,NumVf,0.05*Global.N,mu,(w/wmax)^alpha,Global.evaluated./Global.evaluation,PopDec1,PopObj1,MSE1,V1,A2,A2Obj);
    
    if ~isempty(PopNew)
        [~,ib]= intersect(PopNew,A2.decs,'rows');
        PopNew(ib,:)=[];
        
        if ~isempty(PopNew)
            New       = INDIVIDUAL(PopNew);
        else
            New = [];
        end
    else
        New = [];
    end
    
    A2        = [A2,New];
    %     A1        = UpdataArchive(A2,New,[V;[]],mu,NI);
    A1 = A2;
	

   
    
    
end
end