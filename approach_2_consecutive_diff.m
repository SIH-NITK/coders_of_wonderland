clear
all_files = {};
all_fname = dir(['Clipped_NDVI\awifs_ndvi*clipped.tif']);
diff = [];
diff_sum=[];

for i = 1:48

tmp = imread(['Clipped_NDVI\',all_fname(i).name]);
tmp = im2bw(tmp,maps,0.05); 

% tmp = imadjust(tmp,stretchlim(tmp),[]);

%  imshow(tmp)

all_files(i) = mat2cell(tmp,2135,2118);
end

diff = {};
for i = 1:47
   diff(i) = mat2cell(cell2mat(all_files(i))-cell2mat(all_files(i+1)),2135,2118); 
end

for i = 1:47
%     imshow(cell2mat(diff(i)))
    diff_sum(i) = sum(sum(cell2mat(diff(i))));
end
plot(diff_sum)


