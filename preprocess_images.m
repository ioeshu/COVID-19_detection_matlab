function IMout = preprocess_images(filename)

I = imread(filename);

if ~ismatrix(I) % if matrix is not in form m*n
    I=rgb2gray(I); 
end
IMout = cat(3,I,I,I); % creating matrix in the form of m*n*3
end
