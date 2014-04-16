function icp(source, target)
   % Execute ICP for a given set of source and target points, which are 
   % of the same dimension.

   N = size(source, 2);
   
   % initialize R as identity, t as zero
   R = eye(4);
   t = [0; 0; 0; 0];
   
   % Add number of random trees to the params
   params = struct('algorithm', 'kdtree', 'trees', 10, 'checks', 128);
   % TODO Build index for repeated use?
   
   RMS_changed = 1;
   RMS = rms(source, target(:, 1:N));
   
   while RMS_changed == 1
      
      fprintf('RMS: %d\n', RMS);
      pause
      
      % transform source according to R and t
      transformed_source = R * source + repmat(t, 1, N);
      centroid_source = mean(source,2)
      
      % find closest points using kdtree
      [results, dists] = ...
         flann_search(transformed_source, target, 1, params);
      disp(transformed_source)
      disp(results)
      
      new_RMS = sqrt(sum(dists(:))/N);
      if new_RMS == RMS
          RMS_changed = 0;
          break;
      end
      RMS = new_RMS;
      
      % redefine R and t using SVD
      selected_target = target(:, results);
      disp(selected_target)
      
      centroid_selected_target = mean(selected_target, 2)
      
      cova = (source - repmat(centroid_source, 1, N)) * ...
             (target - repmat(centroid_selected_target, 1, N))';

      [U,~,V] = svd(cova);
      % Find rotation matrix
      R = V * U'
      % Find translation
      t = -R * centroid_source + centroid_selected_target
   end
end
