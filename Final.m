[filename,pathname]=uigetfile('*','select a X-Ray image'); %Browse image
filewithpath=strcat(pathname,filename);
I=imread(filewithpath); 
im = preprocess_images(filewithpath); % preprocess image with the preprocess_images fun
imResized = imresize(im,[224 224]); 
    [class, score]=classify(netTransfer,imResized); % classifaction fun
figure
    imshow(imResized)
    title([ 'Predclass=' char(string(class)),', ','score=',num2str(max(score))])
