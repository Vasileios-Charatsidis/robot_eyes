function main(dirname)
   % Read files etc.
   
   for i=0:2
     filename_source = createfilename(i); 
     filename_target = createfilename(i+1);
     % Read pointcloud files
     source = readPcd(sprintf('%s/%s', dirname, filename_source));
     target = readPcd(sprintf('%s/%s', dirname, filename_target));
     
   end
end