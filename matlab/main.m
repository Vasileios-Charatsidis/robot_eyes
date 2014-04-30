function main(dirname)
   % Read files etc.
   
   %for i=0:2
   %  filename_source = createfilename(i); 
   %  filename_target = createfilename(i+1);
   %  % Read pointcloud files
   %  source = readPcd(sprintf('%s/%s', dirname, filename_source));
   %  target = readPcd(sprintf('%s/%s', dirname, filename_target));
   % 
   %end
   
   % Each column is a point!
   
   pointcloud1 = [1 0 0 ;
                  0 1 0 ;
                  0 0 1 ;
                  1 1 1 ];
   pointcloud2 = [0 -1 0 ;
                  1 0 0 ;
                  0 0 1 ;
                  1 1 1 ];
   %pointcloud1 = vertcat(randn(3, 5), zeros(1, 5));
   %pointcloud2 = vertcat(randn(3, 5), zeros(1, 5));
   icp(pointcloud1, pointcloud2);
end