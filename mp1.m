%% Mini Project 1 - Compressed Sensing with LASSO

%% Read the images and display one
nature = imread("nature.bmp");
fishing_boat = imread("fishing_boat.bmp");
image = double(fishing_boat);
imagesc(image); colormap gray

%% Get the user to input the block size
validBlockSize = 0;
while(~validBlockSize)
    k = input("Input block size: ");
    [xSize, ySize] = size(image);
    xBlocks = xSize/k;
    yBlocks = ySize/k;
    if mod(xSize,k) == 0 && mod(ySize,k) == 0
        validBlockSize = 1;
    else
        disp("Invalid Block Size.")
    end
end

%% Put each block into an array
blockArray = zeros(k,k,xBlocks*yBlocks);
blockIncrement = 1;
for xblockNum = 1:xBlocks
    for yblockNum = 1:yBlocks
        blockArray(:,:,blockIncrement) = image(k*(xblockNum-1)+1:k*xblockNum,k*(yblockNum-1)+1:k*yblockNum);
        blockIncrement = blockIncrement + 1;
    end
end

% %% Reconstruct Image from blocks to make sure you did this correctly
% newImage = zeros(xSize,ySize);
% blockIncrement = 1;
% for xblockNum = 1:xBlocks
%     for yblockNum = 1:yBlocks
%         newImage(k*(xblockNum-1)+1:k*xblockNum,k*(yblockNum-1)+1:k*yblockNum) = blockArray(:,:,blockIncrement);
%         blockIncrement = blockIncrement + 1;
%     end
% end
% figure;imshow(newImage/max(max(newImage)))
%% Sample a random S pixels
S = [10,20,30,40,50];
% S = [10, 30, 50, 100, 150]; % For nature
error = zeros(2,length(S));
errorInc = 1;
recoveredImages = zeros(xSize, ySize, length(S)); % pre median filtered

for s = S % Loop through several different sampled images
    % Estimating -255
    randPixels = -1*ones(k,k,xBlocks*yBlocks);
    for blockNum = 1:xBlocks*yBlocks
        randPixelInd = randperm(k^2, s);
        [randPixelRow, randPixelCol] = ind2sub([k,k],randPixelInd);
        for i = 1:s
            randPixels(randPixelRow(i),randPixelCol(i),blockNum) = blockArray(randPixelRow(i), randPixelCol(i), blockNum);
        end
        randPixels(:,:,blockNum) = randPixels(:,:,blockNum);
    end
    imagesc(randPixels(:,:,1)); colormap gray

    %% Display sampled image
    newImageSampled = zeros(xSize,ySize);
    blockIncrement = 1;
    for xblockNum = 1:xBlocks
        for yblockNum = 1:yBlocks
            newImageSampled(k*(xblockNum-1)+1:k*xblockNum,k*(yblockNum-1)+1:k*yblockNum) = randPixels(:,:,blockIncrement);
            blockIncrement = blockIncrement + 1;
        end
    end
    figure; imshow(newImageSampled/max(max(newImageSampled)));

    %% Create the T matrix
    % Sample the T matrix by rows --> A matrix
    %   If a pixel is -1, then delete that row

    P = k;
    Q = k;

    T = zeros(k*k);
    colCounter = 0;

    for u = 1:k
        for v = 1:k
            % Updates on every change in u and v
            colCounter = colCounter + 1;
            T_col = zeros(P,Q); % Actually a matrix but it will be reshaped
            if u == 1
                a = sqrt(1/P);
            else
                a = sqrt(2/P);
            end
            if v == 1
                b = sqrt(1/Q);
            else
                b = sqrt(2/Q);
            end

            % Calculate the values of each column and then concatenate them all
            % together into one big T matrix
            for x = 1:k
                for y = 1:k
                    T_col(x,y) = a*b*cos(pi*(2*x-1)*(u-1)/(2*P))*cos(pi*(2*y-1)*(v-1)/(2*Q));
                end
            end
            T_col = reshape(T_col,[P*Q,1]);

            T(:,colCounter) = T_col;
        end
    end

    %% Sample T and C for a block to get A and B
    A_mats = zeros(s, k*k, blockNum); % Sampled Transformation Matrix
    B_mats = zeros(s, 1, blockNum); % Sampled Pixels (+1 to 0 DC)
    for blockIndex = 1:blockNum
        block = randPixels(:,:,blockIndex);
        pixCol = reshape(block, [k*k,1]);
        missingPicIndices = find(pixCol == -1);

        % Delete the rows you don't want from T
        T_temp = T;
        T_temp(missingPicIndices,:) = [];
        A_mats(:,:,blockIndex) = T_temp;

        % Delete the rows you don't want from C
        pixCol(missingPicIndices) = [];
        B_mats(:,1,blockIndex) = pixCol;

    end

    %% Determine the value of lambda through cross val and get gamma via lasso
    lambda = logspace(-6, 6, 25);
    % lambda = logspace(-12, 1, 25);
    selectedLambda = zeros(1,blockNum);
    gamma = zeros(k*k, 1, blockNum);

    % Randomly sample pixels from the B matrix into 4 folds
    s
    tic
    for b = 1:blockNum
        % Get the MSE for each lambda through cross validation
        MSE = rsCalcMSE(s, 20, A_mats(:,:,b), B_mats(:,:,b), lambda, k);


        % Find the smallest MSE and the lambda associated with it
        idx = find(MSE == min(MSE));
        idx = idx(1);
        selectedLambda(b) = lambda(idx);

        % Run lasso regression
        [gammaTemp, fitIntercept] = lasso(A_mats(:,2:end,b), B_mats(:,:,b), 'Lambda', selectedLambda(b), "Intercept", true);
        gamma(:,:,b) = [fitIntercept.Intercept*k; gammaTemp]; 
        % times k because we don't want to scale down the constant values

        % Progress Indicator - Prints the Block Number every 50 blocks
%         if(mod(b,50) == 0)
%             b
%         end
    end
    toc

    %% Reconstruct the image
    % Get all the C Matrices

    C_mats = zeros(k, k, blockNum);
    for block = 1:blockNum
        C_mats(:,:,block) = reshape(T*gamma(:,:,block),[k,k]);
    end

    % Loop through the original image and replace the value of any pixel that
    % was originally -1 in randPixels
    % Additionally, calculate the error
    C_mats_replace = randPixels;
    for b = 1:blockNum
        for kx = 1:k
            for ky = 1:k
                if(C_mats_replace(kx, ky, b) == -1)
                    C_mats_replace(kx, ky, b) = C_mats(kx, ky, b);
                end

            end
        end
    end

    % Piece the image back together
    newImage = zeros(xSize,ySize);
    blockIncrement = 1;
    for xblockNum = 1:xBlocks
        for yblockNum = 1:yBlocks
            newImage(k*(xblockNum-1)+1:k*xblockNum,k*(yblockNum-1)+1:k*yblockNum) = C_mats_replace(:,:,blockIncrement);
            blockIncrement = blockIncrement + 1;
        end
    end
    
    % Save the new images
    recoveredImages(:,:,errorInc) = newImage;
    medImage = medfilt2(newImage);
    figure;imagesc(medImage);
    colorbar
    colormap gray
    
    %% Calculate the Error
    sampledErrorMedian = sum(sum((double(image)./255 - medImage).^2));
    sampledError = sum(sum((double(image)./255 - newImage).^2));
    error(1,errorInc) = sampledError/(xSize*ySize);
    error(2,errorInc) = sampledErrorMedian/(xSize*ySize);
    errorInc = errorInc + 1;
    
end

%% Create Plots and Save Images
figure;
plot(S, 255^2*error(1,:), '-s'); hold on
plot(S, 255^2*error(2,:), '-s');
xlabel("Samples");
ylabel("Mean Squared Error");
title("MSE versus Samples");
legend(["Before Median Filtering", "After Median Filtering"]);

% cd natureImages
% for i = 1:length(S)
%     imwrite(recoveredImages(:,:,i), "im"+S(i)+".png");
%     imwrite(medfilt2(recoveredImages(:,:,i)), "im"+S(i)+"_m.png");
% end
% 
% cd ..

%% Create Functions
% createFolds is a function that sorts pixels into a specified number of
% folds. pixIndices is a cell full of lists that have different lengths.
function pixIndices = createFolds(Bmat, numFolds)
    % Initialize pixIndices
    pixIndices = {};
    for i = 1:numFolds
        pixIndices(numFolds) = {[]};
    end
    
    % Randomly shuffle pixels and distribute them into k bins
    sortedPix = randperm(length(Bmat));
    for i = 1:length(sortedPix)
        foldNum = mod(i,numFolds)+1;
        pixIndices(foldNum) = {[pixIndices{foldNum}, sortedPix(i)]};
    end
end

function MSE = calcMSE(foldPix, a_mat, b_mat, numFolds, lambdas)
    MSE = zeros(1,length(lambdas));
    for i = 1:numFolds
        % Sort Data into Testing and Training
        testingPix = foldPix{i};
        trainingPix = [];
        for j = 1:numFolds
            if i ~= j
                trainingPix = [trainingPix, foldPix{j}];
            end
        end
        
        % Get the pixels for the training folds
        a_train = a_mat(:, 2:end); % Don't train with the DC component
        a_train(testingPix, :) = [];
        b_train = b_mat;
        b_train(testingPix, :) = [];
        
        % Get the pixels for the testing folds
        a_test = a_mat(testingPix, :);
        b_test = b_mat(testingPix, :);
        
        [gamma, fitInfo] = lasso(a_train, b_train, 'Lambda', lambdas, "Intercept", true);
        gamma = [fitInfo.Intercept; gamma];
        
        RSS = (b_test - a_test*gamma).^2;
        MSE = MSE + 1/length(b_test) * RSS;
    end
end

function MSE = rsCalcMSE(s, M, a_mat, b_mat, lambdas, k)
    m = floor(s/6);
    MSE = zeros(1,length(lambdas));
    for i = 1:M
        randPix = randperm(length(b_mat));
        testingIdx = randPix(1:m);

        a_train = a_mat(:, 2:end);
        a_train(testingIdx,:) = [];
        b_train = b_mat;
        b_train(testingIdx, :) = [];

        % Get the pixels for the testing folds
        a_test = a_mat(testingIdx, :);
        b_test = b_mat(testingIdx, :);

        [gamma, fitInfo] = lasso(a_train, b_train, 'Lambda', lambdas, "Intercept", true);
        gamma = [fitInfo.Intercept*k; gamma];

        % Calculate the residual sum of squares for the image
        RSS = sum((b_test - a_test*gamma).^2);
        
        % Calculate the mean squared error
        MSE = MSE + (1/length(b_test) * RSS)/M;
    end
    
end