Qubitsトモグラフィーの使い方
==

まず、qubit_tomo.pyを作業Directory(project)下にコピーします。


    .
    └project
        └qubit_tomo.py

そこで

    ~\project>python qubit_tomo.py

と入力します。  

    ------------------------------------------------------------
    PLEASE ENTER NUMBER OF QUBITS
    ------------------------------------------------------------
    >>

と表示されるのでシミュレートしたいQubitsの数を入力してください。  
今回は4つで試します。  
また、測定基底は卒業論文に記載しているとおりです。  
他の基底で測定した場合はcodeを書き換えてください。  

     ------------------------------------------------------------
    PLEASE ENTER NUMBER OF QUBITS
    ------------------------------------------------------------
    >>
    4
    ------------------------------------------------------------
    PLEASE ENTER PATH OF EXPERIMENTAL DATA DIRECTORY

    LIKE THIS >> .\datadirectory
    ------------------------------------------------------------
    >> 

次にtxtファイルで保存した実験データがあるディレクトリパスを入力してください。  
実験データが複数あってもすべてについてトモグラフィーしてくれます。  
ただし、3D描画画面が出たままではシミュレートは進みません。  
注意してください。
実験データを次のように保存している場合は以下のように指定できます。

    .
    └project
        ｜
        ├testdata
        ｜  ｜
        ｜  └test.txt
        ｜
        └qubit_tomo.py


    ------------------------------------------------------------
    PLEASE ENTER PATH OF EXPERIMENTAL DATA DIRECTORY

    LIKE THIS >> .\datadirectory
    ------------------------------------------------------------
    >>
    ./testdata

次に、計算結果の出力先ディレクトリ名を入力してください。  
ここはなんでもいいです。  
何も入力しなければ”default”になります。  
例えば、

    ------------------------------------------------------------
    PLEASE ENTER NAME OF RESULT DIRECTORY

    THE RESULT DATA WILL SAVED AT
    '.\result\qubit\iterative(or poisson)\{ YOUR ENTED DIRECTORY NAME }\{ EXPERIMENTAL DATA FILE NAME }_result.txt'

    IF EMPTY, THE NAME OF RESULT DIRECTORY IS 'default'
    ------------------------------------------------------------
    >>
    test

ここで、疑似実験データを作成するか聞かれます。  
"yes"以外の入力はすべて"no"と判断されます。  
疑似実験データは各実験データの回数をそれぞれ期待値としたポアソン分布に沿うようにランダムに生成されます。  


    ------------------------------------------------------------
    PLEASE ENTER ANSWER WHETHER DO POISSON DISTRIBUTED SIMULATION
    IF YOU DO, PLEASE ENTER 'yes'
    IF YOU ENTER ANOTHER WORD OR EMPTY, YOUR ANSWER IS REGARED AS 'no'
    ------------------------------------------------------------
    >>
    yes
    YOUR ANSWER IS: 'yes'
    ------------------------------------------------------------
    PLEASE ENTER PATHS OF EXPERIMENTAL DATA

    IF THERE ARE MULTIPLE DATA FILE YOU WANT TO TOMOGRAPHY,
    ENTER ALL PATHS SEPARATED WITH SPACE.
    LIKE THIS >> .\datadirectory\ex1.txt .\datadirectory\ex2.txt ...
    ------------------------------------------------------------
    >>

"yes"と入力すると、このように表示されるので、疑似実験データのもととなる実験データ**ファイルパス**を入力してください。  
"no"の場合は最後のの並列化数入力まで飛ばされます。    
"yes"の場合、例えば、

    YOUR ANSWER IS: 'yes'
    ------------------------------------------------------------
    PLEASE ENTER PATHS OF EXPERIMENTAL DATA

    IF THERE ARE MULTIPLE DATA FILE YOU WANT TO TOMOGRAPHY,
    ENTER ALL PATHS SEPARATED WITH SPACE.
    LIKE THIS >> .\datadirectory\ex1.txt .\datadirectory\ex2.txt ...
    ------------------------------------------------------------
    >>
    ./testdata/test.txt

すると、疑似実験データを何パターン生成するか聞かれるので生成したい数を入力してください。  

    ------------------------------------------------------------
    PLEASE ENTER ITERATION TIME OF EACH POISSON SIMULATION
    ------------------------------------------------------------
    >>
    5

最後に、いくつ並列化させて計算するか聞かれます。  
また、現在使用しているパソコンの並列化可能な最大の数が表示されるので参考にしてください。  
（ **※注意：** 並列化可能な最大の数の半分程度でないとCPU使用率が跳ね上がってシミュレートが進まなくなります。シミュレート実行後に、必ずリソースモニターなどでCPU使用率を確認してください。）

    ------------------------------------------------------------
    HOW MANY TIMES DO YOU WANT TO PARALLELIZE?
    IF THE NUMBER IS TOO LARGE, THE PARFORMANCE OF SIMULATION BECOME LOWER.
    THE NUMBER OF LOGICAL PROCESSOR OF YOUR COMPUTER IS >>
    6
    RECOMENDED NUMBER IS LESS THAN THE ABOVE NUMBER.
    ------------------------------------------------------------
    >>
    2

あとは、シミュレートが終わるまで待つだけです。  
上でも述べましたが、3D描画が表示されている間は他のシミュレートは進まないので、必要がない場合はsrcのl.310の

    plotResult(numberOfQubits, estimatedDensityMatrix, baseNames)

を次のようにコメントアウトしてください。

    #plotResult(numberOfQubits, estimatedDensityMatrix, baseNames)


出力結果
--

出力結果は次のディレクトリに保存されます。

    .
    └project
        ｜
        ├result
        ｜  ｜
        ｜  └qubit
        ｜      ｜
        ｜      ├iterative
        ｜      ｜  ｜
        ｜      ｜  └test
        ｜      ｜      ｜
        ｜      ｜      └result.txt
        ｜      ｜
        ｜      └poisson
        ｜          ｜
        ｜          └test
        ｜              ｜
        ｜              └result.txt
        ｜
        ├testdata
        ｜  ｜
        ｜  └test.txt
        ｜
        └qubit_tomo.py

`result.txt`には理想状態とのfidelityが保存されています。  
理想状態は`qubit_tomo.py`に直接用意しています。  
必要に応じて書き換えてください。