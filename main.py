from data_loader import Data, CustomData
from config import NUM_RUNS, TRAIN_DATASETS, TEST_DATASETS, SUBTASK, TASK, CUE_MODEL, SCOPE_MODEL, SCOPE_METHOD, INITIAL_LEARNING_RATE, EPOCHS, PATIENCE
from neg_model import CueModel, ScopeModel

def main():
    bioscope_full_papers_data = Data('bioscope/full_papers.xml', dataset_name='bioscope')
    sfu_data = Data('SFU_Review_Corpus_Negation_Speculation', dataset_name='sfu')
    bioscope_abstracts_data = Data('bioscope/abstracts.xml', dataset_name='bioscope')
    if TASK == 'negation':
        sherlock_train_data = Data('starsem-st-2012-data/cd-sco/corpus/training/SEM-2012-SharedTask-CD-SCO-training-09032012.txt', dataset_name='starsem')
        sherlock_dev_data = Data('starsem-st-2012-data/cd-sco/corpus/dev/SEM-2012-SharedTask-CD-SCO-dev-09032012.txt', dataset_name='starsem')
        sherlock_test_gold_cardboard_data = Data('starsem-st-2012-data/cd-sco/corpus/test-gold/SEM-2012-SharedTask-CD-SCO-test-cardboard-GOLD.txt', dataset_name='starsem')
        sherlock_test_gold_circle_data = Data('starsem-st-2012-data/cd-sco/corpus/test-gold/SEM-2012-SharedTask-CD-SCO-test-circle-GOLD.txt', dataset_name='starsem')

        vetcompass_train_data = Data('vetcompass_subsets/train/negspec', dataset_name='vetcompass')
        vetcompass_dev_data = Data('vetcompass_subsets/dev/negspec', dataset_name='vetcompass')
        vetcompass_test_data = Data('vetcompass_subsets/test/negspec', dataset_name='vetcompass')
    
    cue_list = []
    cont = False
    for i,sent in enumerate(sherlock_test_gold_circle_data.cue_data.cues):
        for j,tok in enumerate(sent):
            if tok == 1:
                if cont == True:
                    cue = cue.strip()
                    cue_list.append(cue)
                    cue = ''
                    cont = False
                cue = sherlock_test_gold_circle_data.cue_data.sentences[i][j]
                cue = cue.strip()
                cue_list.append(cue)
                cue = ''
                cont = False
            elif tok == 2:
                if cont == True:
                    cue+= ' ' + sherlock_test_gold_circle_data.cue_data.sentences[i][j]
                else:
                    cue = sherlock_test_gold_circle_data.cue_data.sentences[i][j]
            elif tok == 3:
                if cont == True:
                    cue = cue.strip()
                    cue_list.append(cue)
                    cue = ''
                    cont = False
    cue_list = set(cue_list)
    for cue in cue_list:
        print(cue)                       
            
    # print(vetcompass_dev_data.cue_data.)

    return 0
    # print('HERE')
    # print(vetcompass_train_data)
    # print(vetcompass_dev_data.cue_data.sentences)
    # print(vetcompass_test_data)
    sent_list = []
    cue_list = []
    # for i,sent in enumerate(vetcompass_dev_data.cue_data.sentences):
    #     sent_text = ' '.join(sent)
    #     sent_list.append(sent_text)
    #     cue_list.append(vetcompass_dev_data.cue_data.cues[i])
    #     # mydata = CustomData(["Hi there this might be good"], cues = [[3,3,3,1,3,3]])
    # custom_data = CustomData(sent_list, cues = cue_list)
    # print(custom_data.sentences[-1])
    # print(custom_data.cues[-1])
    scope_list = []
    for i,sent in enumerate(vetcompass_dev_data.scope_data.sentences):
        sent_text = ' '.join(sent)
        sent_list.append(sent_text)
        cue_list.append(vetcompass_dev_data.scope_data.cues[i])
        # mydata = CustomData(["Hi there this might be good"], cues = [[3,3,3,1,3,3]])
    custom_data = CustomData(sent_list, cues = cue_list)

    # return 0

    for run_num in range(NUM_RUNS):
        first_dataset = None
        other_datasets = []
        if 'sfu' in TRAIN_DATASETS:
            first_dataset = sfu_data
        if 'bioscope_full_papers' in TRAIN_DATASETS:
            if first_dataset == None:
                first_dataset = bioscope_full_papers_data
            else:
                other_datasets.append(bioscope_full_papers_data)
        if 'bioscope_abstracts' in TRAIN_DATASETS:
            if first_dataset == None:
                first_dataset = bioscope_abstracts_data
            else:
                other_datasets.append(bioscope_abstracts_data)
        if 'sherlock' in TRAIN_DATASETS:
            if first_dataset == None:
                first_dataset = sherlock_train_data
            else:
                other_datasets.append(sherlock_train_data)
        if 'vetcompass' in TRAIN_DATASETS:
            if first_dataset == None:
                first_dataset = vetcompass_train_data
            else:
                other_datasets.append(vetcompass_train_data)

        if SUBTASK == 'cue_detection':
            train_dl, val_dls, test_dls = first_dataset.get_cue_dataloader(other_datasets = other_datasets)
            if 'sherlock' in TRAIN_DATASETS:
                val_dls = val_dls[:-1]
                append_dl, _, _ = sherlock_dev_data.get_cue_dataloader(test_size = 0.00000001, val_size = 0.00000001)
                val_dls.append(append_dl)
                test_dls = test_dls[:-1]
                sherlock_dl, _, _ = sherlock_test_gold_cardboard_data.get_cue_dataloader(test_size = 0.00000001, val_size = 0.00000001, other_datasets = [sherlock_test_gold_circle_data])
                test_dls.append(sherlock_dl)

            if 'vetcompass' in TRAIN_DATASETS:
                val_dls = val_dls[:-1]
                append_dl, _, _ = vetcompass_dev_data.get_cue_dataloader(test_size = 0.00000001, val_size = 0.00000001)
                val_dls.append(append_dl)
                test_dls = test_dls[:-1]
                vetcompass_dl, _, _ = vetcompass_test_data.get_cue_dataloader(test_size = 0.00000001, val_size = 0.00000001)
                test_dls.append(vetcompass_dl)

            test_dataloaders = {}
            val_dataloaders = {}
            idx = 0
            if 'sfu' in TRAIN_DATASETS:
                if 'sfu' in TEST_DATASETS:
                    test_dataloaders['sfu'] = test_dls[idx]
                    val_dataloaders['sfu'] = val_dls[idx]

                idx+=1
            elif 'sfu' in TEST_DATASETS:
                sfu_dl, _, _ = sfu_data.get_cue_dataloader(test_size = 0.00000001, val_size = 0.00000001)
                test_dataloaders['sfu'] = sfu_dl
                val_dataloaders['sfu'] = sfu_dl

            if 'bioscope_full_papers' in TRAIN_DATASETS:
                if 'bioscope_full_papers' in TEST_DATASETS:
                    test_dataloaders['bioscope_full_papers'] = test_dls[idx]
                    val_dataloaders['bioscope_full_papers'] = val_dls[idx]

                idx+=1
            elif 'bioscope_full_papers' in TEST_DATASETS:
                bioscope_full_papers_dl, _, _ = bioscope_full_papers_data.get_cue_dataloader(test_size = 0.00000001, val_size = 0.00000001)
                test_dataloaders['bioscope_full_papers'] = bioscope_full_papers_dl
                val_dataloaders['bioscope_full_papers'] = bioscope_full_papers_dl
                
            if 'bioscope_abstracts' in TRAIN_DATASETS:
                if 'bioscope_abstracts' in TEST_DATASETS:
                    test_dataloaders['bioscope_abstracts'] = test_dls[idx]
                    val_dataloaders['bioscope_abstracts'] = val_dls[idx]


                idx+=1
            elif 'bioscope_abstracts' in TEST_DATASETS:
                bioscope_abstracts_dl, _, _ = bioscope_abstracts_data.get_cue_dataloader(test_size = 0.00000001, val_size = 0.00000001)
                test_dataloaders['bioscope_abstracts'] = bioscope_abstracts_dl
                val_dataloaders['bioscope_abstracts'] = bioscope_abstracts_dl

            if 'sherlock' in TRAIN_DATASETS:
                if 'sherlock' in TEST_DATASETS:
                    test_dataloaders['sherlock'] = test_dls[idx]
                    val_dataloaders['sherlock'] = val_dls[idx]

                idx+=1
            elif 'sherlock' in TEST_DATASETS:
                sherlock_dl, _, _ = sherlock_test_gold_cardboard_data.get_cue_dataloader(test_size = 0.00000001, val_size = 0.00000001, other_datasets = [sherlock_test_gold_circle_data])
                test_dataloaders['sherlock'] = sherlock_dl
                val_dataloaders['sherlock'] = sherlock_dl

            if 'vetcompass' in TRAIN_DATASETS:
                if 'vetcompass' in TEST_DATASETS:
                    test_dataloaders['vetcompass'] = test_dls[idx]
                    val_dataloaders['vetcompass'] = val_dls[idx]

                idx+=1
            elif 'vetcompass' in TEST_DATASETS:
                vetcompass_dl, _, _ = vetcompass_test_data.get_cue_dataloader(test_size = 0.00000001, val_size = 0.00000001)
                test_dataloaders['vetcompass'] = vetcompass_dl
                val_dataloaders['vetcompass'] = vetcompass_dl


        elif SUBTASK == 'scope_resolution':
            train_dl, val_dls, test_dls = first_dataset.get_scope_dataloader(other_datasets = other_datasets)
            if 'sherlock' in TRAIN_DATASETS:
                val_dls = val_dls[:-1]
                append_dl, _, _ = sherlock_dev_data.get_scope_dataloader(test_size = 0.00000001, val_size = 0.00000001)
                val_dls.append(append_dl)
                test_dls = test_dls[:-1]
                sherlock_dl, _, _ = sherlock_test_gold_cardboard_data.get_scope_dataloader(test_size = 0.00000001, val_size = 0.00000001, other_datasets = [sherlock_test_gold_circle_data])
                test_dls.append(sherlock_dl)
            if 'vetcompass' in TRAIN_DATASETS:
                val_dls = val_dls[:-1]
                append_dl, _, _ = vetcompass_dev_data.get_scope_dataloader(test_size = 0.00000001, val_size = 0.00000001)
                val_dls.append(append_dl)
                test_dls = test_dls[:-1]
                vetcompass_dl, _, _ = vetcompass_test_data.get_scope_dataloader(test_size = 0.00000001, val_size = 0.00000001)
                test_dls.append(vetcompass_dl)

            test_dataloaders = {}
            val_dataloaders = {}

            idx = 0
            if 'sfu' in TRAIN_DATASETS:
                if 'sfu' in TEST_DATASETS:
                    test_dataloaders['sfu'] = test_dls[idx]
                    val_dataloaders['sfu'] = val_dls[idx]

                idx+=1
            elif 'sfu' in TEST_DATASETS:
                sfu_dl, _, _ = sfu_data.get_scope_dataloader(test_size = 0.00000001, val_size = 0.00000001)
                test_dataloaders['sfu'] = sfu_dl
                val_dataloaders['sfu'] = sfu_dl

            if 'bioscope_full_papers' in TRAIN_DATASETS:
                if 'bioscope_full_papers' in TEST_DATASETS:
                    test_dataloaders['bioscope_full_papers'] = test_dls[idx]
                    val_dataloaders['bioscope_full_papers'] = val_dls[idx]

                idx+=1
            elif 'bioscope_full_papers' in TEST_DATASETS:
                bioscope_full_papers_dl, _, _ = bioscope_full_papers_data.get_scope_dataloader(test_size = 0.00000001, val_size = 0.00000001)
                test_dataloaders['bioscope_full_papers'] = bioscope_full_papers_dl
                val_dataloaders['bioscope_full_papers'] = bioscope_full_papers_dl

            if 'bioscope_abstracts' in TRAIN_DATASETS:
                if 'bioscope_abstracts' in TEST_DATASETS:
                    test_dataloaders['bioscope_abstracts'] = test_dls[idx]
                    val_dataloaders['bioscope_abstracts'] = val_dls[idx]

                idx+=1
            elif 'bioscope_abstracts' in TEST_DATASETS:
                bioscope_abstracts_dl, _, _ = bioscope_abstracts_data.get_scope_dataloader(test_size = 0.00000001, val_size = 0.00000001)
                test_dataloaders['bioscope_abstracts'] = bioscope_abstracts_dl
                val_dataloaders['bioscope_abstracts'] = bioscope_abstracts_dl

            if 'sherlock' in TRAIN_DATASETS:
                if 'sherlock' in TEST_DATASETS:
                    test_dataloaders['sherlock'] = test_dls[idx]
                    val_dataloaders['sherlock'] = val_dls[idx]

                idx+=1
            elif 'sherlock' in TEST_DATASETS:
                sherlock_dl, _, _ = sherlock_test_gold_cardboard_data.get_scope_dataloader(test_size = 0.00000001, val_size = 0.00000001, other_datasets = [sherlock_test_gold_circle_data])
                test_dataloaders['sherlock'] = sherlock_dl
                val_dataloaders['sherlock'] = sherlock_dl

            if 'vetcompass' in TRAIN_DATASETS:
                if 'vetcompass' in TEST_DATASETS:
                    test_dataloaders['vetcompass'] = test_dls[idx]
                    val_dataloaders['vetcompass'] = val_dls[idx]

                idx+=1
            elif 'vetcompass' in TEST_DATASETS:
                vetcompass_dl, _, _ = vetcompass_test_data.get_scope_dataloader(test_size = 0.00000001, val_size = 0.00000001)
                test_dataloaders['vetcompass'] = vetcompass_dl
                val_dataloaders['vetcompass'] = vetcompass_dl


        else:
            raise ValueError("Unsupported subtask. Supported values are: cue_detection, scope_resolution")


        if SUBTASK == 'cue_detection':
            model = CueModel(full_finetuning=True, train=True, learning_rate = INITIAL_LEARNING_RATE)
        elif SUBTASK == 'scope_resolution':
            model = ScopeModel(full_finetuning=True, train=True, learning_rate = INITIAL_LEARNING_RATE)
        else:
            raise ValueError("Unsupported subtask. Supported values are: cue_detection, scope_resolution")
        
        
        model.train(train_dl, val_dls, epochs=EPOCHS, patience=PATIENCE, train_dl_name = ','.join(TRAIN_DATASETS), val_dl_name = ','.join(TRAIN_DATASETS))

        # for k in test_dataloaders.keys():
        #     print(f"Evaluate on {k}:")
        #     # model.evaluate(test_dataloaders[k], test_dl_name = k)
        #     model.evaluate(val_dataloaders[k], test_dl_name = k)
        # custom_data_loader = custom_data.get_cue_dataloader()
        custom_data_loader = custom_data.get_scope_dataloader()

        pred = model.predict(custom_data_loader)
        print(len(pred))
        flat_pred = [item for sublist in pred for item in sublist]

        # print(pred)
        for i,sent in enumerate(vetcompass_dev_data.scope_data.sentences):
            # if flat_pred[i] != vetcompass_dev_data.cue_data.cues[i]:
            print(sent)
            print(flat_pred[i])
            print(vetcompass_dev_data.scope_data.scopes[i])
            print('=============================================')
        print(f"\n\n************ RUN {run_num+1} DONE! **************\n\n")


if __name__ == "__main__":
    main()