function icp()
  
   % initialize R as identity
   R = eye(3);
   

   while RMS_unchanged
      % find closest points using kdtree

      % redefine R and t using SVD

      % recompute RMS
      RMS_new = rms(source, target, mapping); 

   end
end
