function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
%C = 1;
%sigma = 0.3;
C0=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma0=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
%error=zeros(numel(C)*numel(sigma),1);
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
error_old=1000;
error_new=0;
for i=1:numel(C0)
    for j=1:numel(C0)
    model = svmTrain(X, y, C0(i), @(x1, x2) gaussianKernel(x1, x2, sigma0(j)))
    predictions= svmPredict(model, Xval);
   
    error_new=mean(double(predictions ~= yval)); %returns 1 if predictions != yval
   
      if(error_new<error_old)
        error_old=error_new;
        id1=i;
        id2=j ;
     
      end
      
    end
    
%[min_val, id]=min(error);
C=C0(id1);
sigma=sigma0(id2);
% =========================================================================

fprintf ("The minimum error is found for sigma=%f \t and C=%f \t ",sigma,C);
end
end
