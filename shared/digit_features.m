function [P, tiles] = digit_features(filename, expectedRows)
% Extract 35 features per digit from separated rows of dark handwriting.
% Requires blank gaps between rows and between digits (no ruled paper).
I = imread(filename);
I = double(I);
if ndims(I) == 3
    I = mean(I(:,:,1:3),3);
end
lo = min(I(:)); hi = max(I(:));
assert(hi > lo, 'The image is blank or has no contrast.');
I = (I-lo)/(hi-lo);
ink = I < 0.5; % Dark ink = 1; white background = 0.
rowMask = any(ink,2);
edges = diff([false; rowMask; false]);
rowStart = find(edges == 1); rowEnd = find(edges == -1)-1;
assert(length(rowStart) == expectedRows, ...
    'Wrong number of rows: use clean paper and blank gaps between rows.');
P = []; tiles = {};
for row = 1:expectedRows
    line = ink(rowStart(row):rowEnd(row),:);
    colMask = any(line,1);
    edges = diff([false colMask false]);
    colStart = find(edges == 1); colEnd = find(edges == -1)-1;
    for k = 1:length(colStart)
        digit = line(:,colStart(k):colEnd(k));
        occupied = find(any(digit,2));
        digit = digit(occupied(1):occupied(end),:);
        % Simple nearest-neighbor resizing to 70 rows by 50 columns.
        ri = round(linspace(1,size(digit,1),70));
        ci = round(linspace(1,size(digit,2),50));
        tile = digit(ri,ci);
        features = zeros(35,1);
        for a = 1:7
            for b = 1:5
                block = tile((a-1)*10+1:a*10,(b-1)*10+1:b*10);
                features((a-1)*5+b) = mean(block(:));
            end
        end
        P(:,end+1) = features; %#ok<AGROW>
        tiles{end+1} = tile; %#ok<AGROW>
    end
end
end
