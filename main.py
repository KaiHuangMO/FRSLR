from skmultilearn.dataset import load_dataset


from FRSLR import FR3
from sklearn.metrics import f1_score
import mld_metrics


testDataset = ['emotions']

for data in testDataset:
    print ('start' + data)
    X, y, _, _ = load_dataset(data, 'train')
    X_testO, y_test, _, _ = load_dataset(data, 'test')

    X = X.toarray()
    y = y.toarray()

    X_testO = X_testO.toarray()
    y_test = y_test.toarray()

    print("X.shape: " + str(len(X)))
    print("y.shape: " + str(len(y)))
    print("Descriptive stats:")

    # to avoid future warnings for sklearn
    import warnings
    warnings.filterwarnings("ignore")

    X_train = X
    y_train = y

    import copy


    print("X_train.shape: " + str(X_train.shape))
    print("X_test.shape: " + str(X_testO.shape))
    print("y_train.shape: " + str(y_train.shape))
    print("y_test.shape: " + str(y_test.shape))
    print(len(X_train))

    # instantiate the classifier
    from skmultilearn.adapt import MLkNN
    from skmultilearn.adapt import BRkNNaClassifier
    from skmultilearn.ensemble import RakelD, RakelO

    from skmultilearn.problem_transform import BinaryRelevance
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from skmultilearn.problem_transform import ClassifierChain
    from sklearn.ensemble import RandomForestClassifier

    X_test = copy.deepcopy(X_testO)

    for kk in range(0, 4):
        br_clf = MLkNN()
        if kk == 0:
            br_clf = MLkNN()
        if kk == 1:
            br_clf = BinaryRelevance(
                classifier=RandomForestClassifier(),
                require_dense=[False, True]
            )
        if kk == 2:
            br_clf = RakelD(
                base_classifier=RandomForestClassifier(),
                base_classifier_require_dense=[True, True]
            )
            #continue
        if kk == 3:
            br_clf = ClassifierChain(
                classifier=RandomForestClassifier(),
                require_dense=[False, True]
            )
        br_clf.fit(copy.deepcopy(X_train), copy.deepcopy(y_train))
        y_pred = br_clf.predict(X_test)#.toarray()
        print ('-----Oringinal----- ')
        print(str((f1_score(y_test, y_pred, average='macro'))) + '\t' + str((f1_score(y_test, y_pred, average='micro'))))

    from LP_ROS import LP_ROS

    X_mlsmote, Y_mlsmote,_ = LP_ROS(copy.deepcopy(X_train), copy.deepcopy(y_train), p = 25)

    for kk in range(0, 4):
        br_clf = MLkNN()
        if kk == 0:
            br_clf = MLkNN()
        if kk == 1:
            br_clf = BinaryRelevance(
                classifier=RandomForestClassifier(),
                require_dense=[False, True]
            )
        if kk == 2:
            br_clf = RakelD(
                base_classifier=RandomForestClassifier(),
                base_classifier_require_dense=[True, True]
            )
            #continue
        if kk == 3:
            br_clf = ClassifierChain(
                classifier=RandomForestClassifier(),
                require_dense=[False, True]
            )
        print(kk)

        br_clf.fit(X_mlsmote, Y_mlsmote)
        # predict
        y_pred = br_clf.predict(X_test)#.toarray()
        print ('-----LP_ROS-----')
        print("y_train.shape: " + str(Y_mlsmote.shape))
        print(str((f1_score(y_test, y_pred, average='macro'))) + '\t' + str((f1_score(y_test, y_pred, average='micro'))))


    from ML_ROS import ML_ROS

    X_mlsmote, Y_mlsmote,_ = ML_ROS(copy.deepcopy(X_train), copy.deepcopy(y_train), p = 25)
    for kk in range(0, 4):
        br_clf = MLkNN()
        if kk == 0:
            br_clf = MLkNN()
        if kk == 1:
            br_clf = BinaryRelevance(
                classifier=RandomForestClassifier(),
                require_dense=[False, True]
            )
        if kk == 2:
            br_clf = RakelD(
                base_classifier=RandomForestClassifier(),
                base_classifier_require_dense=[True, True]
            )
            #continue
        if kk == 3:
            br_clf = ClassifierChain(
                classifier=RandomForestClassifier(),
                require_dense=[False, True]
            )
        print(kk)

        br_clf.fit(X_mlsmote, Y_mlsmote)
        # predict
        y_pred = br_clf.predict(X_test)  # .toarray()
        print('-----ML_ROS-----')
        print("y_train.shape: " + str(Y_mlsmote.shape))
        print(str((f1_score(y_test, y_pred, average='macro'))) + '\t' + str((f1_score(y_test, y_pred, average='micro'))))


    from REMEDIAL import REMEDIAL

    X_mlsmote, Y_mlsmote = REMEDIAL(copy.deepcopy(X_train), copy.deepcopy(y_train))
    for kk in range(0, 4):
        br_clf = MLkNN()
        if kk == 0:
            br_clf = MLkNN()
        if kk == 1:
            br_clf = BinaryRelevance(
                classifier=RandomForestClassifier(),
                require_dense=[False, True]
            )
        if kk == 2:
            br_clf = RakelD(
                base_classifier=RandomForestClassifier(),
                base_classifier_require_dense=[True, True]
            )
            #continue
        if kk == 3:
            br_clf = ClassifierChain(
                classifier=RandomForestClassifier(),
                require_dense=[False, True]
            )
        print(kk)

        br_clf.fit(X_mlsmote, Y_mlsmote)
        # predict
        y_pred = br_clf.predict(X_test)  # .toarray()
        print('-----REMEDIAL-----')
        print("y_train.shape: " + str(Y_mlsmote.shape))
        print(str((f1_score(y_test, y_pred, average='macro'))) + '\t' + str((f1_score(y_test, y_pred, average='micro'))))



    from MLSMOTE import MLSMOTE

    X_mlsmote, Y_mlsmote, y_new_ori, min_set = MLSMOTE(copy.deepcopy(X_train), copy.deepcopy(y_train), 10)


    for kk in range(0, 4):
        br_clf = MLkNN()
        if kk == 0:
            br_clf = MLkNN()
        if kk == 1:
            br_clf = BinaryRelevance(
                classifier=RandomForestClassifier(),
                require_dense=[False, True]
            )
        if kk == 2:
            br_clf = RakelD(
                base_classifier=RandomForestClassifier(),
                base_classifier_require_dense=[True, True]
            )
            #continue
        if kk == 3:
            br_clf = ClassifierChain(
                classifier=RandomForestClassifier(),
                require_dense=[False, True]
            )
        print(kk)
        if kk == 0:
            print('MLSMOTE: ' + str(mld_metrics.mean_ir(Y_mlsmote)) + ' ' + str(mld_metrics.cvir(Y_mlsmote)))

        br_clf.fit(X_mlsmote, Y_mlsmote)
        # predict
        y_pred = br_clf.predict(X_test)  # .toarray()
        print('-----MLSMOTE-----')
        print(str((f1_score(y_test, y_pred, average='macro'))) + '\t' + str((f1_score(y_test, y_pred, average='micro'))))


    X_mix2, Y_mix2 = FR3(X_mlsmote, Y_mlsmote, y_new_ori, min_set, copy.deepcopy(X_train), copy.deepcopy(y_train))
    for kk in range(0, 4):
        br_clf = MLkNN()
        if kk == 0:
            br_clf = MLkNN()
        if kk == 1:
            br_clf = BinaryRelevance(
                classifier=RandomForestClassifier(),
                require_dense=[False, True]
            )
        if kk == 2:
            br_clf = RakelD(
                base_classifier=RandomForestClassifier(),
                base_classifier_require_dense=[True, True]
            )
            #continue
        if kk == 3:
            br_clf = ClassifierChain(
                classifier=RandomForestClassifier(),
                require_dense=[False, True]
            )
        print(kk)
        if kk == 0:
            print('FR MLSMOTE3: ' + str(mld_metrics.mean_ir(Y_mix2)) + ' ' + str(mld_metrics.cvir(Y_mix2)))

        br_clf.fit(X_mix2, Y_mix2)
        # predict
        y_pred = br_clf.predict(X_test)  # .toarray()
        print('-----FR MLSMOTE R3-----')
        #print("y_train.shape: " + str(Y_mix2.shape))
        print(str((f1_score(y_test, y_pred, average='macro'))) + '\t' + str((f1_score(y_test, y_pred, average='micro'))))

    from MLSOL import MLSOL
    mlsol = MLSOL(perc_gen_instances=0.25, k=10)
    X_mlsmote, Y_mlsmote, y_new_ori, _ = mlsol.fit_resample(copy.deepcopy(X_train), copy.deepcopy(y_train), gen_num=0)
    for kk in range(0, 4):

        br_clf = MLkNN()
        if kk == 0:
            br_clf = MLkNN()
        if kk == 1:
            br_clf = BinaryRelevance(
                classifier=RandomForestClassifier(),
                require_dense=[False, True]
            )
        if kk == 2:
            br_clf = RakelD(
                base_classifier=RandomForestClassifier(),
                base_classifier_require_dense=[True, True]
            )
            #continue
        if kk == 3:
            br_clf = ClassifierChain(
                classifier=RandomForestClassifier(),
                require_dense=[False, True]
            )
        print(kk)
        br_clf.fit(X_mlsmote, Y_mlsmote)
        # predict
        y_pred = br_clf.predict(X_test)  # .toarray()
        print('-----MLSOL-----')
        #print("y_train.shape: " + str(Y_mlsmote.shape))
        if kk == 0:
            print('MLSOL: ' + str(mld_metrics.mean_ir(Y_mlsmote)) + ' ' + str(mld_metrics.cvir(Y_mlsmote)))

        print(str((f1_score(y_test, y_pred, average='macro'))) + '\t' + str((f1_score(y_test, y_pred, average='micro'))))

    X_mix2, Y_mix2 = FR3(X_mlsmote, Y_mlsmote, y_new_ori, min_set, copy.deepcopy(X_train), copy.deepcopy(y_train))
    for kk in range(0, 4):
        br_clf = MLkNN()
        if kk == 0:
            br_clf = MLkNN()
        if kk == 1:
            br_clf = BinaryRelevance(
                classifier=RandomForestClassifier(),
                require_dense=[False, True]
            )
        if kk == 2:
            br_clf = RakelD(
                base_classifier=RandomForestClassifier(),
                base_classifier_require_dense=[True, True]
            )
            #continue
        if kk == 3:
            br_clf = ClassifierChain(
                classifier=RandomForestClassifier(),
                require_dense=[False, True]
            )
        print(kk)
        if kk == 0:
            print('FR MLSOL3: ' + str(mld_metrics.mean_ir(Y_mix2)) + ' ' + str(mld_metrics.cvir(Y_mix2)))

        br_clf.fit(X_mix2, Y_mix2)
        # predict
        y_pred = br_clf.predict(X_test)  # .toarray()
        print('-----FR MLSOL3-----')
        #print("y_train.shape: " + str(Y_mlsmote.shape))

        print(str((f1_score(y_test, y_pred, average='macro'))) + '\t' + str((f1_score(y_test, y_pred, average='micro'))))

