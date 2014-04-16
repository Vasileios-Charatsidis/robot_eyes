function name = createfilename(int)
    if int < 10
        name = sprintf('0000000%u', int);
    elseif int < 100
        name = sprintf('000000%u', int);
    else
        printf('Filename invalid');
        exit;
    end
end