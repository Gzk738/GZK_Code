import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
normal  = [0.0, 0.36934424519061726, 0.7865015530900777, 0.22204112011659607, 0.11920515655192801, 0.09462263205821493, 0.02147414889374691, 0.15215481449843912, -0.022641897957084184, 0.02589882269927251, 0.04437515869217518, -0.01117306874987402, 0.004421799614159154, -0.020556872136471966, 0.007859373115165144, 0.022086179651511256, 0.0, 0.01575296830906432, 0.010590090445756553, -0.00915689558940503, 0.004520738776437335, -0.003016080462645726, 0.006493097662964489, -0.012142644444250155, -0.002966876728571428, 0.00018182168119658713, -0.029374073068767572, -0.010480782585641502, -0.013514963744025712, -0.01713499329094535, -0.0016652641516516952, -0.010607711858325656, -0.010131524341456543, 0.0009442768130926894, -0.006698255997896712, -0.017318931045879574, -0.00723336247612875, -0.005715960884618394, 0.002563503894158933, -0.013915227187076157, -0.011966101833433872, 0.000523109249966915, -0.009699247929792957, -0.03267236725998104, -0.009896857856484061, -0.013490584197728613, -0.0020804792111991247, -0.00678289586057064, -0.009294042650396107, -0.011828308346380125, -0.0014327125238531035, -0.002023103812301482, 3.040664923837758e-05, -0.014637384198114623, -0.02557032364633229, -0.008750745385743062, -0.024791349527560257, 0.011616860077306322, -0.01784455769339346, 0.0072317648008063075, -0.0014655690196481806, -0.030910432867951636, 0.008206204938486485, -0.005400858394069574, -0.007308363395668892, -0.017553806975842582, -0.01437378863408514, -0.00686640159343674, 0.012504346957281894, -0.005786287237247589, -0.0072793913327336675, -0.007508838924845244, -0.007559870742461803, -0.0036873486708976517, -0.0075135849333144876, 0.0017471793104014627, 0.0013798112645130533, -0.007578228000214514, -0.03715414924197748, -0.02229368131470494, -0.03706538760236994, -0.026716274864315537, -0.009074658933959924, -0.002704392439262253, -0.019840377155242298, -0.02557809198630179, -0.013436822000216345, -0.034281137813616906, -0.01876300442774596, -0.005750472302249845, -0.015169315902985455, -0.04765964028231051, -0.012918236409759663, 0.010057104840119607, -0.008883039174341587, 0.02289223137825933, 0.012333797344820392, 0.017196353679169666, 0.018031252972399863, 0.01778324430788121, 0.0018399667739549437, 0.018983449017295356, -0.003707954352232239, 0.026546364527890078, 0.015173731709051742, 0.021994768944474333, 0.02540082621352449, 0.004413427943235553, 0.009862028479647815, 0.039059629052404184, 0.0017403058453367654, 0.020465109282050132, -0.001032639205913417, 0.006834746871400258, 0.017683848379989906, 0.004865572735951688, 0.014445425434034499, 0.03611511019921474, 0.017627627577981305, 0.009390662966727643, 0.011367141696047344, 0.045126168036411805, 0.03793750426900198, 0.035120631438574634, 0.01757514540649001, 0.029715449740634284, 0.027740924328697035, -0.001885166269864968, 0.060599174947539886, 0.08740727572855991, 0.11918765419688139, 0.08200028075277144, 0.0435541418583289, 0.011016186790227677, 0.025841917958272082, 0.0157933284693579, 0.01131169947348107, 0.009944882992239498, 0.047802138592884805, -0.059090738703634536, 0.047083295679362945, 0.036659022934647546, 0.01709419884873614, 0.060812873400694537, 3.650359106612669e-06, 0.005666092212682797, 0.014811644057052565, 0.00759515042354804, 0.010982383006694922, -0.011676026314788415, 0.0023346275522459444, 0.002722028519232467, -0.004319334425548456, 0.016215827278007307, 0.0039034960647977646, 0.014458275697345032, -0.007108239876160085, 0.0021484722625122097, 0.01806763261991003, -0.01802211992154078, -0.015172935617700008, 0.01069542237541425, -0.008926039601723031, -0.003938028946828912, -0.008577241373252276, -0.008907573927062394, -0.021662172078901636, -0.00514343306008453, -0.02972486466997568, -0.02146477755747749, 0.0018422580582834601, 0.022978026714908634, -0.144186411743607, -0.04087462277610429, -0.1863706950635808, 0.0]

i = 0
a = []


a = gaussian_filter(normal, sigma=5)
plt.plot(range(len(a)), a, label = "sigma = 5" )
plt.xlabel('all tokenizers')
plt.ylabel('attribution')
plt.title("interpretation analysis")
plt.legend()
plt.show()


