from main_train import *

TEST_SIZE = 512
TEST_BATCH = 16
IMAGES_PER_ID = IMGS_PER_ID
#MAX_DUO_COUNT = (IMAGES_PER_ID*TEST_BATCH)*(IMAGES_PER_ID*TEST_BATCH)
if __name__ == "__main__":

    print("Loading dataset")

    #test_dataset_path = r'./img_align_celeba_full/eval_partition/1_identity_augment'
    #  comment / un comment to choose dataset_path for TRAIN ERROR or test_dataset_path for TEST ERROR
    test_dataset = CustomDataset(root_dir=test_dataset_path, transform=transform, test=False, size=TEST_SIZE)
    #test_dataset = CustomDataset(root_dir=test_dataset_path, transform=transform, test=False, size=TEST_SIZE)

    test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH, shuffle=False, num_workers=4)
    print("Dataset loaded")
    print("Loading saved model")

    # model = MobileFaceNet(embedding_size=EMBEDDING_DIM).to(device)
    # # load pre-trained weights for MobileFaceNet
    # load_model_path = 'Epoch_17.pt'
    # state_dict_loaded = model.state_dict()
    # #state_dict_pretrained = torch.load(load_model_path, map_location=device)
    # state_dict_pretrained = torch.load(load_model_path, map_location=device)['state_dict']
    # state_dict_temp = {}
    #
    # # state_dict = torch.load(load_model_path)['state_dict']
    # # model.load_state_dict(state_dict)
    #
    # for k in state_dict_loaded:
    #     if 'bn.num_running_batches' not in k:
    #         state_dict_temp[k] = state_dict_pretrained['backbone.'+k]
    #     else:
    #         print(k, 'not loaded!')
    # state_dict_loaded.update(state_dict_temp)
    # model.load_state_dict(state_dict_loaded)
    # del state_dict_loaded, state_dict_pretrained, state_dict_temp



    model = models.resnet50(pretrained=True).to(device)
    #model = ResNetEmbedding(embedding_dim=EMBEDDING_DIM).to(device)

    #model = torch.load("./models/cosinecustomYanMODEL512_batch16_total20000_epochs40_lr0.0001_margin0.3_time2024-11-19@13-53-02.pth").to(device)
    #model = torch.load("./models/hyperMODEL128_batch16_total20000_epochs40_lr0.0001_margin0.2_time2024-11-05@10-51-45.pth").to(device)
    #model = torch.load("./models/cosinecustomMODEL512_batch16_total20000_epochs25_lr0.0001_margin0.2_time2024-11-17@15-55-55.pth").to(device)
    #model = torch.load("./models/cosinecustomMODEL128_batch20_total20000_epochs20_lr1e-05_margin0.2_time2024-11-10@06-22-16.pth").to(device)
    #model = torch.load("./models/hopeMODEL512_batch16_total20000_epochs20_lr0.05_margin0.2_time2024-11-04@08-01-56.pth").to(device)
    #model = torch.load("./models/hopeMODEL512_batch16_total20000_epochs40_lr0.05_margin0.2_time2024-11-04@10-51-13.pth").to(device)

    model.eval()  # Set the model to evaluation mode (important for inference)
    print("model loaded")


    with torch.no_grad():  # Disable gradient calculations
        model_true_actual_false = 0
        model_false_actual_true = 0
        model_false_actual_false = 0
        model_true_actual_true = 0
        duos_counter = 0
        same_duos_counter = 0
        diff_duos_counter = 0
        # p_same_data = np.empty(MAX_DUO_COUNT)
        # p_diff_data = np.empty(MAX_DUO_COUNT)
        p_same_data = []
        p_diff_data = []
        m_distances = []
        matches = []
        batch_cnt = 0

        print("Initiating test")
        for identities, labels in test_dataloader:
            batch_cnt += 1
            print("Current batch number: ", batch_cnt)
            identities_filtered = [id for id in identities if len(id) == IMGS_PER_ID]
            images = torch.stack(list(chain.from_iterable(identities_filtered)))
            labels_filtered = [lbl for lbl in labels if len(lbl) == IMGS_PER_ID]
            all_labels = torch.stack(list(chain.from_iterable(labels_filtered)))

            images, matching_labels = images.to(device), all_labels.to(device)

            # Forward pass to get embeddings
            images_embed = model(images)  # Output shape: (batch_size, EMBEDDING_DIM)
            # Ensure images and labels are of the same length
            assert len(images_embed) == len(matching_labels)
            print("images in batch: ", len(images_embed))

            # Generate all unique duos (2-tuples) of images and corresponding labels
            duos = itertools.combinations(range(len(images_embed)), 2)

            # Iterate over the trios and get the corresponding images and labels
            for idx1, idx2 in duos:
                duos_counter += 1
                img1_l = matching_labels[idx1]
                img2_l = matching_labels[idx2]
                # print("no sq", images_embed[idx1])
                # print("yes sq", images_embed[idx1].unsqueeze(0))
                #dist = F.pairwise_distance(images_embed[idx1].unsqueeze(0), images_embed[idx2].unsqueeze(0))  # hope this does euclid dist
                #dist = F.cosine_similarity(images_embed[idx1].unsqueeze(0),images_embed[idx2].unsqueeze(0))
                dist = DISTANCE_FUNC(images_embed[idx1].unsqueeze(0), images_embed[idx2].unsqueeze(0))  # hope this does compute currectly, maybe need to add .unsqeeze(0)
                dist = dist.item()
                m_distances.append(dist)
                matches.append(int(img1_l == img2_l))
        #####
        # perform accuracy tests bcz we already have data for it...
        m_distances = np.array(m_distances, dtype='float32')
        matches = np.array(matches, dtype='int32')
        # print("dists:", distances[:20])
        # print("matches", matches[:20])
        p_same_data = m_distances[matches == 1]
        p_diff_data = m_distances[matches == 0]
        ROC_labels = matches
        ROC_data = m_distances

        plt.figure(figsize=(10, 6))

        sns.histplot(p_same_data, bins=50, kde=True,
                     color='green',
                     stat='density')  # kde=True adds the KDE curve
        sns.histplot(p_diff_data, bins=50, kde=True,
                     color='red',
                     stat='density')  # kde=True adds the KDE curve

        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title("Empirical Distribution (Histogram with KDE)")
        #plt.show() show at the end

        # Compute the ROC curve
        #### I assume that your verification code is correct based on what I have seen, so I do not add anything else to it

        fpr, tpr, thresholds = roc_curve(ROC_labels, ROC_data)
        ## ## change to EER calculation
        eer, min_index = compute_eer(fpr,tpr)  # closer to 0 is better! 50% is worst
        optimal_threshold = thresholds[min_index]
        print("Optimal Threshold:", optimal_threshold)
        if DISTANCE_FUNC_NAME == 'euc':
            model_true_actual_false = len(
                p_diff_data[p_diff_data < optimal_threshold])
            model_false_actual_true = len(
                p_same_data[p_same_data >= optimal_threshold])
            model_false_actual_false = len(
                p_diff_data[p_diff_data >= optimal_threshold])
            model_true_actual_true = len(
                p_same_data[p_same_data < optimal_threshold])
        if DISTANCE_FUNC_NAME == 'cosine':
            model_true_actual_false = len(
                p_diff_data[p_diff_data >= optimal_threshold])
            model_false_actual_true = len(
                p_same_data[p_same_data < optimal_threshold])
            model_false_actual_false = len(
                p_diff_data[p_diff_data < optimal_threshold])
            model_true_actual_true = len(
                p_same_data[p_same_data >= optimal_threshold])

        model_correct = model_true_actual_true + model_false_actual_false
        true_catch = model_true_actual_true / (
                model_true_actual_true + model_false_actual_true)
        false_catch = model_false_actual_false / (
                model_false_actual_false + model_true_actual_false)
        print("True catch rate: ", true_catch)
        print("False catch rate: ", false_catch)
        print("Total accuracy [correct/total]: ",
              model_correct / duos_counter)
        print("EER ", eer)
        #####

        plt.show()

