import numpy as np
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
 
# # Data
# df=pd.DataFrame({'x': range(1,11), 'y1': np.random.randn(10), 'y2': np.random.randn(10)+range(1,11), 'y3': np.random.randn(10)+range(11,21) })
 
# # multiple line plot
# plt.plot( 'x', 'y1', data=df, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
# plt.plot( 'x', 'y2', data=df, marker='', color='olive', linewidth=2)
# plt.plot( 'x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")
# plt.legend()

# import matplotlib.pyplot as plt
# import numpy as np
 
# # create data
# values=np.cumsum(np.random.randn(1000,1))
 
# # use the plot function
# plt.plot(values)

# plt.show()

# libraries and data
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import datetime
 
# # Make a data frame
# df=pd.DataFrame({'x': ["1 parallel", "6 parallel"], '(|0000>+|1111>)(<0000|+<1111|) + |0001><0001| + |0010><0010| + |0100><0100| + |1000><1000|': [, datetime.0:14:33.761596], '(|0000>+|1111>)(<0000|+<1111|)': , '(|0001> + |0010> + |0100> + |1000>)(<0001| + <0010| + <0100| + <1000|)': , 'all': })
 
# # style
# plt.style.use('seaborn-darkgrid')
 
# # create a color palette
# palette = plt.get_cmap('Set1')
 
# # multiple line plot
# num=0
# for column in df.drop('x', axis=1):
#     num+=1
#     plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)
 
# # Add legend
# plt.legend(loc=2, ncol=2)
 
# # Add titles
# plt.title("A (bad) Spaghetti plot", loc='left', fontsize=12, fontweight=0, color='orange')
# plt.xlabel("Time")
# plt.ylabel("Score")


# import matplotlib.pyplot as plt
# from matplotlib import dates as mdates
# from datetime import datetime as dt

# ylist=['1:24:18.172140', '0:14:33.761596']

# # xlist,ylistを datetime型に変換
# xlist = ["1 parallel", "6 parallel"]
# ylist = [dt.strptime(d, '%H:%M:%S.%f') for d in ylist]
# # データをプロット
# ax = plt.subplot()
# ax.scatter(xlist,ylist)
# # Y軸の設定 (目盛りを１時間毎,範囲は 0:00～23:59とする)
# ax.yaxis.set_major_locator(mdates.HourLocator())
# ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
# ax.set_ylim([dt.strptime('00:00','%H:%M'), dt.strptime('02:00','%H:%M')])

# plt.show()

# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import pandas as pd
# import io

# # グラフ作成
# plt.figure()
# plt.plot(['1:24:18', '0:14:33'])

# # 時刻のフォーマットを設定
# yfmt = mpl.dates.DateFormatter("%H:%M:%S")
# plt.gca().yaxis.set_major_formatter(yfmt)

# plt.show()

# import matplotlib.pyplot as plt
# import pandas as pd

# timels = ['1:24:18', '0:14:33']
# timedata = pd.to_timedelta(['0' + time for time in timels])

# # 複数の折れ線グラフをプロット：線の色（color）・点のマーカー（marker）・線の種類（linetype）・凡例表示のためのラベル（label）でグラフを区別
# plt.plot([1,6], , color = 'red', marker = 'o', linestyle = '-', label = 'Sensor1')

# yfmt = mpl.dates.DateFormatter("%H:%M:%S")
# plt.xlabel('parallel')                  # x軸のラベル
# plt.ylabel('time')                     # y軸のラベル
# plt.legend(loc = 'upper right')         # 凡例を右上（upper right）に表示

# plt.show()                              # 図の表示

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime as dt
from matplotlib.dates import DateFormatter

fig = plt.figure()

x = np.array([1,6])

timels = ['1:24:18', '0:14:33']
timels = [dt.strptime(d, '%H:%M:%S') for d in timels]
y = np.array(timels)
ax1 = fig.add_subplot(1, 1, 1)
ax1.plot(x, y, label='case 1', marker='o')
# timels = ['1:24:55', '0:15:19']
# timels = [dt.strptime(d, '%H:%M:%S') for d in timels]
# y = np.array(timels)
# ax1.plot(x, y, label='estimated state')
# plt.legend(loc=2, ncol=2)
# ax1.yaxis.set_major_formatter(DateFormatter('%H:%M'))
# ax1.set_xlabel('parallel')
# ax1.set_ylabel('calculation time')
# # ax1.set_title('(|0001> + |0010> + |0100> + |1000>)(<0001| + <0010| + <0100| + <1000|)')
# plt.xticks(np.arange(1, 7, 5), labels=[1,6])

timels = ['0:04:04', '0:00:40']
timels = [dt.strptime(d, '%H:%M:%S') for d in timels]
y = np.array(timels)
# ax2 = fig.add_subplot(1, 2, 1)
ax1.plot(x, y, label='case 2', marker='o')
# ax2.yaxis.set_major_formatter(DateFormatter('%H:%M'))
# ax2.set_xlabel('parallel')
# ax2.set_ylabel('time')
# ax2.set_title('(|0000>+|1111>)(<0000|+<1111|)')
# plt.xticks(np.arange(1, 7, 5), labels=[1,6])

timels = ['2:38:26', '0:25:13']
timels = [dt.strptime(d, '%H:%M:%S') for d in timels]
y = np.array(timels)
# ax3 = fig.add_subplot(2, 1, 1)
ax1.plot(x, y, label='case 3', marker='o')
# ax3.yaxis.set_major_formatter(DateFormatter('%H:%M'))
# ax3.set_xlabel('parallel')
# ax3.set_ylabel('time')
# ax3.set_title('all')
# plt.xticks(np.arange(1, 7, 5), labels=[1,6])

timels = ['1:40:27', '0:17:16']
timels = [dt.strptime(d, '%H:%M:%S') for d in timels]
y = np.array(timels)
# ax4 = fig.add_subplot(2, 2, 1)
ax1.plot(x, y, label='case 4', marker='o')
# ax4.yaxis.set_major_formatter(DateFormatter('%H:%M'))
# ax4.set_xlabel('parallel')
# ax4.set_ylabel('time')
# ax4.set_title('(|0000>+|1111>)(<0000|+<1111|) + |0001><0001| + |0010><0010| + |0100><0100| + |1000><1000|')
# plt.xticks(np.arange(1, 7, 5), labels=[1,6])

timels = ['0:19:04', '0:03:13']
timels = [dt.strptime(d, '%H:%M:%S') for d in timels]
y = np.array(timels)
# ax5 = fig.add_subplot(3, 1, 1)
ax1.plot(x, y, label='case 5', marker='o')
# ax5.yaxis.set_major_formatter(DateFormatter('%H:%M'))
# ax5.set_xlabel('parallel')
# ax5.set_ylabel('time')
# ax5.set_title('random')

plt.legend(loc=1, ncol=2)
ax1.yaxis.set_major_formatter(DateFormatter('%H:%M'))
ax1.set_xlabel('parallel')
ax1.set_ylabel('calculation time')

plt.xticks(np.arange(1, 7, 5), labels=[1,6])
plt.show()