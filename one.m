% Clear workspace
clear,clc, close all
% Images Datapath – You can modify your path accordingly 
datapath='dataset';
% Image Datastore
imds=imageDatastore(datapath, ...  
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); %Datastore for image data
All_samples=countEachLabel(imds) %Count files in ImageDatastore labels
[imdsTrain,imdsTest] = splitEachLabel(imds,.8,'randomized'); %Split ImageDatastore labels by proportions
net=resnet50;
    lgraph = layerGraph(net) % Extract all layers
        lgraph.Layers % visualize all layers
    lgraph.Layers(175).OutputSize % get number of classes in resnet-50
    clear net;
numClasses = numel(categories(imdsTrain.Labels)) %Number of array elements
 newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    lgraph = replaceLayer(lgraph,'fc1000',newLearnableLayer);
    %A softmax layer applies a softmax function to the input.
    newsoftmaxLayer = softmaxLayer('Name','new_softmax');
    lgraph = replaceLayer(lgraph,'fc1000_softmax',newsoftmaxLayer);
    %A classification layer computes the cross entropy loss for multi-class
   %  classification problems with mutually exclusive classes.
    newClassLayer = classificationLayer('Name','new_classoutput');
    lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',newClassLayer);
    imdsTrain.ReadFcn = @(filename)preprocess_images(filename);
    imdsTest.ReadFcn = @(filename)preprocess_images(filename);
    options = trainingOptions('adam',...
        'MaxEpochs',5,'MiniBatchSize',20,...
        'Shuffle','every-epoch', ...
        'InitialLearnRate',1e-4, ...
        'Verbose',false, ...
        'Plots','training-progress'); %Options for training deep learning neural network
    augmenter = imageDataAugmenter( ...
        'RandRotation',[-5 5],'RandXReflection',1,...
        'RandYReflection',1,'RandXShear',[-0.05 0.05],'RandYShear',[-0.05 0.05]);
    auimds = augmentedImageDatastore([224 224],imdsTrain,'DataAugmentation',augmenter); 
