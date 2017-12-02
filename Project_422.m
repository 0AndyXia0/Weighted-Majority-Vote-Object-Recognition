%% Load image data
% This assumes you have a directory: 101_objectcategories (from caltech
% 101)
% with each scene in a subdirectory
imds = imageDatastore('.\101_objectcategories',...
    'IncludeSubfolders',true,'LabelSource','foldernames')             
%% Equalize number of images of each class in training set
tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category
%Use splitEachLabel method to trim the set.
imds.ReadFcn = @(loc)readAndPreprocessImage(imread(loc));
imds.Labels = categorical(imds.Labels);
imds = splitEachLabel(imds, minSetCount);

[training_set, test_set] = splitEachLabel(imds, 0.7,'randomize');

% Convert labels to categoricals and set read function
training_set.Labels = categorical(training_set.Labels);
training_set.ReadFcn = @(loc)readAndPreprocessImage(imread(loc));
test_set.Labels = categorical(test_set.Labels);
test_set.ReadFcn = @(loc)readAndPreprocessImage(imread(loc));
categories = tbl.Label;
%% Create Visual Vocabulary For Full Set

imds_bag = bagOfFeatures(imds,...
    'VocabularySize',250,'PointSelection','Detector');
%%
imds_data = double(encode(imds_bag, imds));
%% Create Visual Vocabulary For Training Set

train_bag = bagOfFeatures(training_set,...
    'VocabularySize',250,'PointSelection','Detector');
%%
train_data = double(encode(train_bag, training_set));

%% Create Visual Vocabulary for Test Set

test_bag = bagOfFeatures(test_set,...
    'VocabularySize',250,'PointSelection','Detector');
%%
test_data = double(encode(test_bag, test_set));

%% Extract Class Values
train_class = training_set.Labels;
test_class = test_set.Labels;
imds_class = imds.Labels;

imds_data_table = array2table(imds_data);
imds_data_table.classType = imds_class;
train_data_table = array2table(train_data);
train_data_table.classType = train_class;
test_data_table = array2table(test_data);
test_data_table.classType = test_class;
test_data_table.Properties.VariableNames = train_data_table.Properties.VariableNames;
VariableNames1 = train_data_table.Properties.VariableNames;
imds_data_table.Properties.VariableNames = VariableNames1;
test_data_table(:,:) = [];
train_data_table(:,:) = [];
train_data_table.Properties.VariableNames = VariableNames1;
test_data_table.Properties.VariableNames = VariableNames1;

for i = 1:height(imds_data_table)
    if(mod(i,31) < 22)
        train_data_table = [train_data_table; imds_data_table(i,:)];
    else
        test_data_table = [test_data_table; imds_data_table(i,:)];
    end
end

%% Train models & Print Model Accuracies
fprintf('this portion of the program may take a long time. It trains 7 models\n and took roughly 25 minutes on an i7-6700 3.4GHZ processor')
fprintf('If it is running too \nslowly for your machine, you can comment out\n this section, and Uncomment the')
fprintf('section \nlabeled "uncomment this" (highlight and ctrl+T) \nso that you can get to the Majority-Vote section') 
[LinearDiscriminant, LinearDiscriminant_acc] = trainClassifier_LinearDiscriminant(train_data_table);
LinearDiscriminant_acc
[LinearSVM, LinearSVM_acc] = trainClassifier_LinearSVM(train_data_table);
LinearSVM_acc
[QuadraticSVM, QuadraticSVM_acc] = trainClassifier_QuadraticSVM(train_data_table);
QuadraticSVM_acc
[CubicSVM, CubicSVM_acc] = trainClassifier_CubicSVM(train_data_table);
CubicSVM_acc
[FineKNN, FineKNN_acc] = trainClassifier_FineKNN(train_data_table);
FineKNN_acc
[CosineKNN, CosineKNN_acc] = trainClassifier_CosineKNN(train_data_table);
CosineKNN_acc
[EnsembleSSD, EnsembleSSD_acc] = trainClassifier_EnsembleSSD(train_data_table);
EnsembleSSD_acc

%% Uncomment This
% If you are eperiencing long model-train times on your machine, uncomment
% the code below, and comment out previous section

% [LinearDiscriminant, LinearDiscriminant_acc] = trainClassifier_LinearDiscriminant(train_data_table);
% 
% 
% LinearSVM = LinearDiscriminant;
% QuadraticSVM = LinearDiscriminant;
% CubicSVM = LinearDiscriminant;
% FineKNN = LinearDiscriminant;
% CosineKNN = LinearDiscriminant;
% EnsembleSSD = LinearDiscriminant;
% 
% LinearDiscriminant_acc
% LinearSVM_acc =  LinearDiscriminant_acc
% QuadraticSVM_acc =  LinearDiscriminant_acc
% CubicSVM_acc =  LinearDiscriminant_acc
% FineKNN_acc =  LinearDiscriminant_acc
% CosineKNN_acc =  LinearDiscriminant_acc
% EnsembleSSD_acc =  LinearDiscriminant_acc

%% Predict Models
yfit_LinearDiscriminant = LinearDiscriminant.predictFcn(test_data_table);
yfit_LinearSVM = LinearSVM.predictFcn(test_data_table);
yfit_QuadraticSVM = QuadraticSVM.predictFcn(test_data_table);
yfit_CubicSVM = CubicSVM.predictFcn(test_data_table);
yfit_FineKNN = FineKNN.predictFcn(test_data_table);
yfit_CosineKNN = CosineKNN.predictFcn(test_data_table);
yfit_EnsembleSSD = EnsembleSSD.predictFcn(test_data_table);

%% Weighted Majority Vote

predictions = [yfit_LinearDiscriminant, yfit_LinearSVM, yfit_QuadraticSVM, yfit_CubicSVM, yfit_FineKNN,yfit_CosineKNN,yfit_EnsembleSSD];
vote_weights = [LinearDiscriminant_acc,LinearSVM_acc,QuadraticSVM_acc,CubicSVM_acc,FineKNN_acc,CosineKNN_acc,EnsembleSSD_acc]


final_votes = [];
for c = 1:length(predictions)
    votes = [];
    weights = [];
    
    for m = 1:7
        found = -1;
        if(m == 1)
            votes = [votes, predictions(c,m)];
            weights = [weights, vote_weights(m)];
        end
        votecounts = numel(votes);
        for idx = 1:votecounts
            if (votes(idx) == predictions(c,m))
                found = idx;
                weights(found) = weights(found) + vote_weights(m);
            end
            if((idx == numel(votes)) && (found == -1))
                found = idx+1;
                votes = [votes, predictions(c,m)];
                weights = [weights, vote_weights(m)];
            end  
        end
    end
    [val1, idx1] = max(weights);
    final_votes = [final_votes, votes(idx1)];   
end
final_votes;

%% Vote accuracy count
correct_count = 0;
h = categorical(test_set.Labels);
for i = 1:length(final_votes)
    if(final_votes(i) == h(i))
        correct_count = correct_count + 1;
    end
end
fprintf('weighted majority vote model:')
WMV_accuracy = correct_count/length(final_votes)
%% Visualize Feature Vectors 
img = read(training_set);
featureVector = encode(imds_bag, img);

subplot(4,2,1); imshow(img);
subplot(4,2,2); 
bar(featureVector);title('Visual Word Occurrences');xlabel('Visual Word Index');ylabel('Frequency');

img = read(training_set);
featureVector = encode(imds_bag, img);
subplot(4,2,3); imshow(img);
subplot(4,2,4); 
bar(featureVector);title('Visual Word Occurrences');xlabel('Visual Word Index');ylabel('Frequency');

img = read(training_set);
featureVector = encode(imds_bag, img);
subplot(4,2,5); imshow(img);
subplot(4,2,6); 
bar(featureVector);title('Visual Word Occurrences');xlabel('Visual Word Index');ylabel('Frequency');

img = read(training_set);
featureVector = encode(imds_bag, img);
subplot(4,2,7); imshow(img);
subplot(4,2,8); 
bar(featureVector);title('Visual Word Occurrences');xlabel('Visual Word Index');ylabel('Frequency');

%% 