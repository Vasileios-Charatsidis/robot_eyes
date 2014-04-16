function icp(source, target)
   % Execute ICP for a given set of source and target points, which are 
   % of the same dimension.
   %

   % initialize R as identity
   R = eye(3);
   
   RMS_changed = 1;
   while RMS_changed == 1 do 
      % find closest points using kdtree
       
      
      % redefine R and t using SVD
      N = size(source, 1);
      centroid_source = mean(source);
      centroid_target = mean(target); 
      
      cova = (source - repmat(centroid_source, N, 1))' * ...
             (target - repmat(centroid_target, N, 1));
      [U,~,V] = svd(cova);
      % Find rotation matrix
      R = V * U';
      % Find translation
      t = -R * centroid_source' + centroid_target';

      % recompute RMS
      RMS_new = rms(source, target, mapping); 
      if RMS_new == RMS
          RMS = RMS_new;
          RMS_changed = 0;
      end
   end
end
