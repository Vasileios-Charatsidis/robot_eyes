function RMS = rms(A1, A2, phi)
   % Compute RMS given 2 pointclouds
   % A1 - base
   % A2 - target
   % phi - mapping from A1 to A2

   % use mapping 
   corresponding_A2 = A2(phi);
   % compute RMS
   RMS = sqrt(MSE(A1, corresponding_A2))
   
end
