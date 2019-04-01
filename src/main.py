import sys
import preProcessDataSets

if len(sys.argv) > 1 and sys.argv[1] == '--preProcessData':
    print('Preprocessing UCF CC 50 dataset ...')
    preProcessDataSets.preProcessUCFCC50()
    print('Preprocessing ShanghaiTech A dataset ...')
    preProcessDataSets.preProcessShanghaiTechA()
    print('Preprocessing ShanghaiTech B dataset ...')
    preProcessDataSets.preProcessShanghaiTechB()

