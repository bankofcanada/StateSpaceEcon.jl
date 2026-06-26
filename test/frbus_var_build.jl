##################################################################################
# This file is part of StateSpaceEcon.jl
# BSD 3-Clause License
# Copyright (c) 2020-2025, Bank of Canada
# All rights reserved.
##################################################################################

# ---------------------------------------------------------------
# FRBUS_VAR transcribed onto the ModelBaseEcon surface. Mechanically
# converted from TutorialsEcon.jl/4.FRB-US/models/FRBUS_VAR.jl.
#
# Conversions from the legacy DSL:
#  - module/Model() wrapper          -> build_frbus_var() + ModelDef
#  - model.substitutions = true      -> dropped (no aux-var
#                                       substitution)
#  - export heaviside / heaviside def -> dropped (heaviside is a
#                                       built-in registered function)
#  - @autoshocks model _a            -> @autoshocks m _a (also creates
#                                       the autoexogenize pairs)
# @d / @dlog / @log / max / min / exp / log / ifelse are all supported.
# ---------------------------------------------------------------
"""
Build the FRBUS_VAR model on the ModelBaseEcon surface and return
the frozen `Model`. 367 variables (284 endogenous + 83 exogenous),
284 auto-shocks, 284 equations.
"""
function build_frbus_var()
    m = ModelDef(:FRBUS_VAR)
    @parameters m begin
        y_dmptlur = [25.0]
        y_dmptpi = [-25.0]
        y_dpadj = [1.0]
        y_dpgap = [0.1036498839384806, 0.3410415470273469, 0.08204590812944115, 0.03932014681002743, 0.05247494124140634, 0.05248394123132865, 0.328983631621969]
        y_ebfi = [0.0735815097920017, 0.5222641043182303, 0.3022170849291267, 1.0, 0.3835146156808873]
        y_ec = [0.7310605131, 0.157421136, 0.1115183509]
        y_ecd = [0.1561499403562654, -0.05999277989382532, 1.0, 11.87185840275941]
        y_ech = [0.0005280507960180215, -0.005044873655834186, 0.7183359768373699]
        y_ecnia = [0.735, 0.1055, 0.1595]
        y_eco = [0.1584215605793791, 0.4118129978512995, 1.0, 0.3075237404993435]
        y_egfe = [-0.0008045327442607825, -0.1204156945139518, -0.153308131863314, -0.1035937599294917, 1.50381543495264, -0.0009835524480447813, 0.0007256812123007961]
        y_egfet = [-0.3101, -0.1, 1.0]
        y_egfl = [0.0002417378554669395, -0.0795584254267964, 0.2415347480637823, -0.06133379374138964, 1.206872311101357, -0.002507254010777311, 0.002350674896424264]
        y_egflt = [-0.375978, -0.1, 1.0]
        y_egse = [-0.0003205558585065158, -0.1310861864392932, 0.1315799717205844, 0.002629649907733779, 0.9287008343314473, 0.001580665878764003, -0.0008537660921935765]
        y_egset = [-0.32, -0.1, 1.0]
        y_egsl = [0.0003682770277494773, -0.1114151303946257, 0.1562467669415557, 0.02155815410956755, 0.7449614274122569, -0.001432565493088224, 0.001765173794440745]
        y_egslt = [-0.259779, -0.1, 1.0]
        y_eh = [0.01309931436163227, 0.370727139760127, 0.206060944066708, 1.0, -0.05620618046332654, 0.02793327971442171]
        y_emo = [0.01620546980581163, -0.1807104996815989, 1.358186927724613, 1.637077186962015, 0.3188307105827468, 0.4046942138546456]
        y_emp = [0.048026, 0.022115]
        y_ex = [0.8398526775741691, -0.1077278728634364, 1.481642245331822, 1.027448537481708, 1.015857050459678]
        y_fgdpt = [-0.458264, -0.1, 1.0]
        y_fpi10 = [0.6279749057317539, 0.3720250942682461, 0.3221458278398038]
        y_fpi10t = [0.95, 0.05]
        y_fpic = [2.027196898567632, 0.6788298801616212, 0.3211701198383788]
        y_fpxr = [0.048, 0.5]
        y_fpxrr = [0.02744058231523356, 0.2110896761767876]
        y_frl10 = [0.03753227212139513, -0.07704648128878298, 0.06550476702266197, 0.369056454239088, 0.1245511812496244]
        y_frs10 = [0.0, 1.0, 1.0, 0.5, 1.0]
        y_frstar = [0.95, 0.05]
        y_fxgap = [1.290723676327916, -0.4680091148746248, -0.05, 0.03734559019022718]
        y_gtrd = [0.862481931486, -0.000176387604876, -0.00143470943]
        y_hgemp = [0.9, 0.1]
        y_hgpkir = [0.9, 0.1]
        y_hks = [0.4543136031, 0.4847950762, 0.0608913208]
        y_hmfpt = [0.055, 0.95]
        y_hqlfpr = [0.0, 0.95]
        y_hqlww = [0.95, -0.3129029344874886]
        y_huqpct = [0.0, 0.95]
        y_huxb = [-0.01613974358626877, 0.95]
        y_jccan = [0.053902, 0.821408]
        y_ki = [0.01469206254903638, 0.4567399290262596, 0.236031927797704, 0.3072281431760363, -0.00120624751419073]
        y_leo = [0.7566675970336151, -0.01642583348237834]
        y_lfpr = [0.5676074828293328, -0.0008751892020969236]
        y_lhp = [0.2022897898011354, 0.2028806748568088, 0.3720641848854363, 0.6279358151145638, -0.127396041937203]
        y_lww = [0.195710350653204, 0.3184816471961936]
        y_mei = [1.0]
        y_mep = [1.0]
        y_mfpt = [0.0]
        y_pcdr = [-0.003334518696912942, 0.5098481943420772]
        y_pcer = [0.2488609533651416]
        y_pcfr = [-0.1753151166404203, -5.59454305651988e-5, 0.3855084844618753, 0.02031018771733943, 0.3388841893424357, 0.3176985579727371]
        y_pchr = [0.0006138065838894166, 0.5980639856669541]
        y_pcor = [-0.1436, -0.217]
        y_phouse = [0.005158779390135185, 0.9018869955145716, -0.01156922658993683]
        y_phr = [0.0]
        y_picnia = [0.0399, 0.0726]
        y_picxfe = [0.671147437396, 0.00191485982247, 0.98]
        y_pieci = [0.684663089102, -0.0167232072082, 0.000957429911235, 0.98]
        y_pipxnc = [0.462801, 0.229745, -0.284477, 0.1312355937]
        y_pmo = [0.0, 0.3776815983707077, 0.2343966603329429]
        y_poilr = [-0.2189945671343387, -0.007197716577611038, 0.3903451978006103, 0.3940619235700209]
        y_ptr = [0.9, 0.05, 0.05]
        y_pxp = [0.6469, 0.3531]
        y_pxr = [0.0]
        y_qec = [0.3653279067017037, 0.493602432836492, 0.1858248429134642, 0.1873934907158843]
        y_qecd = [-0.5843539676290892, 2.4132166616749, -0.5843539676290892]
        y_qeco = [-0.3258628580621771]
        y_qeh = [1.909300384501687, -0.1168105360328174]
        y_qkir = [-0.001885366737710053]
        y_qpl = [1.0]
        y_qpmo = [-0.003347]
        y_qpxb = [0.0, 1.0]
        y_qpxnc = [2.98507462687, -1.98507462687]
        y_qynidn = [-0.9155533588082586, 0.3548225925232601, 1.0]
        y_rbbbp = [1.972218, -0.189051, 0.848879]
        y_rcar = [2.078274063162198, -1.163656496933583, -0.008285302552441101, 0.6967481719141809, 0.1016693350391095, 0.2015824930467097]
        y_rcgain = [-0.4097125100056558, 0.4243575136593637, 0.2257857751190805, -0.4097125100056558, 0.4243575136593637]
        y_reqp = [2.903727, 0.808086, 0.795819]
        y_rffalt = [0.0551, 1.2, -0.39, 0.6954, -0.5168, 0.3287]
        y_rffintay = [0.5, 1.0, 0.85]
        y_rfftay = [0.5, 1.0]
        y_rfftlr = [-0.5, 0.375, 1.1]
        y_rfynic = [0.2442546132781247, -0.1404162075757351, 0.1444243609858357, 0.6315036756054792]
        y_rfynil = [0.1769720233111774, -0.2484056734487245, 0.08033864767698434, 0.09023943103709647, 0.04450523477009356, 0.1328188190918866, 0.0876033907073424, 0.2614346003836546, 0.0179349568621618]
        y_rg10p = [1.050750815840056, -0.4606588068717281, 0.2287218644237361, 0.9201040880647524]
        y_rg30p = [1.420673758140307, -0.6248294677066961, 0.1349942505218234, 0.9381086057081572]
        y_rg5p = [0.6912438755770502, -0.3495644809998469, 0.9022132931202672]
        y_rgfint = [0.86, 0.003726285406154968]
        y_rgw = [0.00495, 0.00271, 0.00129, 0.00105]
        y_rme = [0.515870426113512, 0.6286405556840244, 0.2555601487902845, 0.2889775740676342, 0.05071546489585005]
        y_rrtr = [0.97, 0.03]
        y_rstar = [0.05]
        y_rtb = [-0.05102548358978556, 0.7997187921520414, 0.1113735515796583, 0.7701225626671688, -0.6812149063988686]
        y_trci = [0.810247648208, 0.00706626139452]
        y_trp = [1.0, 0.603942358608, 0.236576213581, 0.000630587773923]
        y_trpt = [0.05, 0.5, -0.1]
        y_ugfsrp = [0.065666, 0.947688]
        y_uqpct = [0.0]
        y_uxbt = [0.0]
        y_uynicpnr = [-0.07396, 0.779183]
        y_wpon = [0.4106759138763171]
        y_xbo = [1.324705489432872]
        y_xbt = [0.725, 0.275]
        y_xbtr = [0.95]
        y_xfs = [0.6849, 0.0386, 0.1324, 0.0429, 0.0223, 0.0395, 0.0691, 0.1203, -0.1399, -0.0101]
        y_xgdp = [0.9985, 0.6264, -0.6249]
        y_xp = [0.6526679404, 0.0361108836, 0.11825695358, 0.04216893278, 0.0365822346, 0.114213055]
        y_yhpcd = [0.05375]
        y_ynidn = [0.1072695065113163, -0.2095634314107719, 1.0]
        y_ynirn = [0.007593, 0.944044, 0.074817]
        y_zdivgr = [1.453758371220758e-14, -0.03432097699759745, 0.01503760188801399, 0.008139233986467266, 0.02806260424571279, -0.1147631155931414, 0.02378759550862853, -0.08000091281298219, 0.05291584088244318, 0.1180605920154127, -0.0169184631226911, -0.3193168668820861, 0.1900611332314869, 0.05865254915210798, 0.05204047323627287, 0.01513617207696648, 0.002907781394121129, 0.001820586788429011, -0.000832768187365904, 0.9809682279278705]
        y_zebfi = [-1.836241195103586e-16, -0.0004311442119548698, -0.0005071417360301063, -3.881819160880394e-5, 0.0001679875754395107, -0.0009752514829426216, 0.0004172696850180997, 9.804022481480237e-6, 0.0004025448938499502, 0.0001456328815931833, 0.0008091165641542618, 0.0006914817407124366, -0.001524629901125206, 0.0001821021224146923, 0.0001709602428972692, 0.01429456576547675, 0.01004233676573099, 0.005011839434648968, 0.003887436951496095, 0.1422818153960116, -0.1422818153960531]
        y_zecd = [-0.0004244330449108246, -0.000566112732916264, -0.0004278354154853733, 4.275450618659433e-6, -0.001333637468405136, 0.001785102754320539, -0.0002714744059753036, 0.0004596118643773856, 0.0004286088490690454, -0.001112480888047013, 3.611331309391688e-5, 7.975907057934714e-5, 0.001414105742693884, -0.0006396027443177932, -0.0001084142645104688, 0.0002103631242011464, 0.0001780616641343914, 0.0001469127491672944, -0.0001398804267539158, -3.380075732963865e-5, 0.0001669757937064113, 0.0001135069368207403, 0.0001241231278850025, -0.0002035919714857632, 5.798918819304864e-5, 0.0001142807758711503, 1.020354953788767, -0.752244722798112, 0.030859810575484, 0.0288465643523214, -0.007704793774557718, -0.01236375512939394]
        y_zeco = [-7.52202049495858e-5, -7.944069331813561e-5, -2.059316996140204e-5, 0.0001004397794984203, 2.128321856982191e-5, 1.703531535880844e-5, 5.501237638104223e-5, 3.68085672111148e-5, -0.0006301710369219651, 0.0002738755865138929, 0.0001330197561306166, -3.466191405309875e-5, 7.481428873071481e-5, -0.0001301394775207934, -0.000574849476125713, 0.000315791553755267, 0.0003970054362973829, 2.606365933680952e-5, -6.065913885267499e-5, -5.861516974911094e-6, 4.608692992415466e-5, -3.675339093791547e-5, 0.0002055017720241992, -0.0002409377143989024, -0.0001318122876593039, -8.998120362836394e-5, 0.4694132786802296, 0.07322397247249939, 0.02932374759217591, 0.00714849213690236, 0.00907784707291976]
        y_zeh = [-7.825906447294479e-5, -5.221776500989767e-5, -5.11617748264036e-5, 1.061188019850692e-5, 0.0001524702142702607, 0.0001154619164109844, 6.94775905299206e-5, 6.915284271997218e-6, 0.0005016959759548423, -0.0005314894632374061, -0.0001527167228830226, 4.422297252718166e-5, 0.0001710267241107452, -0.0003443250054832286, 5.547462567554765e-5, 4.078231691046518e-5, -3.078619404779462e-5, -4.584642539780123e-6, 7.136415206693365e-6, 1.470739436481251e-5, 3.830847423082347e-5, 3.271967551086854e-5, -6.992854917184709e-5, 4.550096611877513e-7, 5.504613762415508e-5, 2.864192627076078e-5, 0.4277729262319505, 0.004266303023854793, -0.001472171716551593, -0.003348261838736116, -0.004008022551322911]
        y_zgap05 = [1.680034256141885e-14, -0.135018393340065, -0.033843245261391, -0.0391222647986449, 0.0173522186659534, -0.3753303057576422, 0.09700115982970414, 0.0739366214990041, 0.02583604947764563, 0.1785564749514165, 0.190631684734112, 0.8244238037691352, -0.2862444942364248, -0.06697780461567028, -0.08505399389084824]
        y_zgap10 = [-1.72233727697999e-15, -0.07186017182762353, -0.0175779452476843, -0.02011214475127172, 0.00932019093665923, -0.1947856648478703, 0.05145463389633801, 0.03639872748995771, 0.01608667022522785, 0.09084563323655222, 0.1002300708898616, 0.4164381390044698, -0.1384702815508384, -0.03509009429541885, -0.04485907444527642]
        y_zgap30 = [1.233586304082598e-14, -0.03828909910973519, -0.009397163294017394, -0.01076603951015025, 0.004952700162140002, -0.1039011522662461, 0.02741531710257839, 0.01963869374286508, 0.008399118041939092, 0.04844802337909579, 0.05349960175170328, 0.2233011674563412, -0.07470857907468761, -0.0187522007846854, -0.02392742740702374]
        y_zgapc2 = [-0.01418483319860699, -0.00438957847118337, -0.006089864990632572, 0.001274535866759684, -0.0426889990257948, 0.007759946050458921, 0.01912856687923288, -0.002207952775923244, 0.1943845369683129, -0.07640072342643053, -0.01132460234847898, -0.01555183396619014, 0.02338974079366426, 0.01800843887202016]
        y_zlhp = [-0.0002023213774340967, -6.54709155555966e-5, -0.0001720246830139474, 3.13937564958239e-5, -0.001047476022547437, 0.0002598839064585707, 0.0005079095801600575, -4.19862687487686e-5, 0.0003216688046775128, 0.0004084232195080358, -0.005754964729358652, 0.005816827692544119, -0.0002034859294983795, -0.0002715434817101584, 0.6857946901749924, 0.1113240674326222]
        y_zpi10 = [0.04357427172865281, 0.01313050267242623, 0.01553540483390203, 0.001463442508125493, -0.1039822705900788, -0.01432543074446008, -0.02962517842600198, -0.007038183618428771, 0.154971063378944, 0.9262963782569013, 0.06032284838878826, -0.01019384116585764, 0.02546215714985389, 0.02470109220233158]
        y_zpi5 = [0.08178762749631197, 0.02218684188675673, 0.02501945218256163, -9.007062448078559e-5, -0.1456765471759016, -0.03113773606787808, -0.02949319295743922, -0.0275798582145864, 0.2338873344157305, 0.8710961490588724, 0.174192252057006, -0.07184023126885179, 0.04066371951581198, 0.04494462398507848]
        y_zpib5 = [-1.26476978296206e-13, 0.09034003814843485, 0.03168080238087264, 0.02770787918164321, -0.01510178193123456, -0.1969518961104264, -0.05959604539330121, -0.04862163727886338, -0.03073950807678016, 0.335909086859382, 0.7000747567151794, 0.1468903171096267, -0.03885798005146663, 0.05539395398596192, 0.04005203158941733, 0.07826113766639138, 0.03750285709064176, 0.01611932032188939, 0.0334149904260995]
        y_zpic30 = [1.044280285832078e-13, 0.04015650846972114, 0.006934744993613762, 0.008188906123754724, 0.0007885180169860036, -0.05503875125078834, -0.007557417372706724, -0.015808419274298, -0.00357920111984304, 0.08198378901761255, 0.943931322395943, 0.03119549195915805, -0.004756369295027224, 0.01348922660681493, 0.01299481953814399]
        y_zpic58 = [0.3593365920920064, 0.05403868424326922, 0.06039898003529536, -0.01698560834120338, 0.05517476076748708, -0.1153492438678267, -0.004952391392669846, -0.1104437552716417, 0.1755706297646033, 0.5432113519706085, 0.1617613655842677, -0.1753961497534604, 0.08374788806947354, 0.1284899640287538]
        y_zpicxfe = [0.380818884672, 0.00113182715476, 0.00146351917605, 0.00225729733693, 0.0460967342223, 0.0338772671906, 0.0228924215171, 0.0112105032823, -0.0140156100481, 0.0011222896601, 0.00760121840982, -0.00299260406007, 0.0470383710002, -0.0278318348119, -0.00506170904133, -0.00225028901719, 0.00828470603822, 0.500251545448, 11.937795061, 6.84395376806e-5, -0.114076926212, -0.00383816812034, -0.000695300677346]
        y_zpieci = [-0.026022539351, 0.00320414216918, 0.00402676215955, 0.00650489050087, 0.202430424141, 0.196252633802, 0.195837958296, 0.0246831983934, -0.0328787076454, 0.00135903909754, 0.0229838005541, -0.00862383586105, 0.148708914616, -0.0777266665551, -0.0137748704693, -0.00648469451174, 0.0171597038548, 0.393082529889, -4.49541220961, 0.000154587412169, 0.380795785368, -0.0172443115476, -0.00416159167724]
        y_zrff10 = [2.378638428839734e-13, 0.0001217535546263044, -0.00917622608986857, -0.01586120496103595, -0.01154696442869298, 0.2209177910482749, -0.01580133905291904, 0.1303085244646027, -0.06613814788532775, 0.7307131714250785, 0.0364626419250845, 0.3614091512807817, -0.273261010083981, -0.03776285146860941, -0.007077405433229054]
        y_zrff30 = [1.711459357335906e-13, 0.0001806188860987932, -0.004921864186060088, -0.0085101379490877, -0.006220981183951381, 0.11864291327276, -0.008553403609128176, 0.06994575055112089, -0.03560722072285206, 0.8555719605078006, 0.01947236443310801, 0.1938808633160732, -0.146844106705379, -0.02021989944869175, -0.003727736282464759]
        y_zrff5 = [2.601031452674343e-13, 0.006240324440737379, -0.0131975418361076, -0.02478131515494538, -0.01906124455815977, 0.356356038809717, -0.02968927450559823, 0.2137924427116254, -0.1108065994879449, 0.57034739247192, 0.05079977710860725, 0.6152802763768458, -0.4530320275590888, -0.05887447593039786, -0.00911820553312758]
        y_zyh = [8.403034216398837e-5, 0.0007023129900742578, 0.0005988312485600414, 0.00046906873529373, -0.002112400975583133, 0.0001656602252725924, -0.0007780393043584456, 0.0002906813774138189, 0.0009057626374626405, 0.002687425542518095, 0.0006823597205465298, -0.001256388702977828, -0.001854243316194259, 0.002434098677388386, 0.004161665385029935, 0.0007501557850814375, -0.0001921235465779609, -0.0003602829076187046, 0.9999999999999565]
        y_zyhp = [0.0008637359958914742, 0.001043993924986531, 0.0009412688975392932, 0.000458916642717137, -0.001544432205587824, -0.0006407725735247019, -0.000877216982815245, -4.075028532698682e-5, -0.002036934910319444, 0.003079022876028106, 0.001533670579319286, -0.0009794433698078087, -0.003307915461364551, 0.003103172047474443, 0.004220697583687526, 0.0001307621492671122, 5.342070875074808e-5, -0.0008349442433696925, 0.002251478648554251, 0.0002014050026037725, -0.0004274985452560566, -0.0002520786346265581, 0.9999999999999468]
        y_zyhpst = [0.05]
        y_zyhst = [0.05]
        y_zyht = [-0.000334830912493238, 0.0004734682680155963, 0.0002798072585528634, 0.000327960879431146, -0.002500279763979537, 0.0008855681526303969, -0.001214881268111102, 8.524329709881323e-5, 0.002172842548066125, 0.00313028920932204, 0.002114206871939033, 0.0002481445696860439, -0.0007464054935481565, 0.002744349582394016, 0.002708245687651875, 0.0006793788749861876, 0.0003089861693518611, -6.935116601571522e-5, 0.001837608113293888, 0.0005096499176820247, 9.470842999385746e-5, 0.00024263888389989, 0.9999999999999932]
        y_zyhtst = [0.05]
        y_zynid = [5.206855405139395e-16, -0.0001020778460719867, 0.0003486952052516843, 0.0002522503063284875, 0.0002059799369104374, 0.000352649102886596, -0.0009117152893301089, -0.0008332522818027586, 0.0002220282051739009, 0.001170290263072301, -0.0007048476024184937, -0.005098524468651768, 0.001661127416240023, 0.0006345562668170024, 0.001396819940195254, 0.002519112167658877, 0.01551673287897806, 0.01239948969603592, -0.006974010098893951, 1.186101463663362]
    end

    @variables m begin
        # Identity variables:
        "Federal funds rate, first diff" delrff
        "Price inflation aggregation discrepancy" dpgap
        "Investment in equipment, current \$" @log ebfin
        "Personal consumption expenditures, current \$ (NIPA definition)" @log ecnian
        "Federal Government expenditures, current \$" @log egfen
        "Federal government employee compensation, current \$" @log egfln
        "S&L Government expenditures, current \$" @log egsen
        "S&L government employee compensation, current \$" @log egsln
        "Residential investment expenditures" @log ehn
        "Change in private inventories, cw 2012\$" ei
        "Imports of goods and services, cw 2012\$" @log em
        "Imports of goods and services, current \$" @log emn
        "Imports of goods and services ex. petroleum" @log emon
        "Petroleum imports, current \$" @log empn
        "Exports of goods and services, current \$" @log exn
        "US current account balance, current \$" fcbn
        "US current account balance residual, current \$" fcbrn
        "Foreign aggregate GDP (world, bilateral export weights)" @log fgdp
        "Gross stock of claims of US residents on the rest of the world, current \$" @log fnicn
        "Gross stock of liabilities of US residents to the rest of the world, current \$" @log fniln
        "Net stock of claims of US residents on the rest of the world, current \$" fnin
        "Net stock of claims of US residents on the rest of the world, residual" fnirn
        "Foreign aggregate consumer price (G39, import/export trade weights)" @log fpc
        "Nominal exchange rate (G39, import/export trade weights)" @log fpx
        "Corporate taxes paid to rest of world, current \$" @log ftcin
        "Gross investment income received from the rest of the world, current \$" @log fynicn
        "Gross investment income paid to the rest of the world, current \$" @log fyniln
        "Net investment income received from the rest of the world, current \$" fynin
        "Federal government debt stock, current \$" @log gfdbtn
        "Federal government debt stock held by the public, current \$" @log gfdbtnp
        "Government payments" @log gfexpn
        "Federal government net interest payments, current \$" @log gfintn
        "Government receipts and residual" @log gfrecn
        "Federal government budget surplus, current \$" gfsrpn
        "Growth rate of GDP, cw 2012\$ (annual rate)" hggdp
        "Trend growth rate of XGDP, cw 2012\$ (annual rate)" hggdpt
        "Trend growth rate of price of business investment (relative to PXB)" hgpbfir
        "Growth rate of real after-tax corporate profits" hgynid
        "Trend growth rate of output per hour" hlprdt
        "Trend rate of growth of XB  , cw 2012\$ (annual rate)" hxbt
        "Consumption of fixed capital, consumer durables" @log jkcd
        "Capital stock - BFI, 2012\$" @log kbfi
        "Stock of consumer durables, cw 2012\$" @log kcd
        "Stock of residential structures, cw 2012\$" @log kh
        "Capital services, 2012\$" @log ks
        "Government civilian employment ex. gov. enterprise" leg
        "Civilian employment (break adjusted)" leh
        "Employment in  business sector (employee and  self-employed)" lep
        "Potential employment in  business sector" leppot
        "Civilian labor force (break adjusted)" lf
        "Trend labor productivity" lprdt
        "Price index for personal consumption expenditures, cw (NIPA definition)" @log pcnia
        "Consumer price index,total" @log pcpi
        "Consumer price index,excluding food and energy" @log pcpix
        "Price index for GDP, cw" @log pgdp
        "Price index for federal government employee compensation, cw" @log pgfl
        "Price index for S&L government employee compensation, cw" @log pgsl
        "Four-quarter percent change in PCE prices" pic4
        "Four-quarter percent change core in PCE prices" picx4
        "Inflation rate, GDP, cw" pigdp
        "Ratio of price of BFI stock (KBFI) to PXP" pkbfir
        "Compensation per hour,  business" @log pl
        "Price index for petroleum imports" @log pmp
        "Price of imported oil (\$ per barrel)" @log poil
        "Price index for business output" @log pxb
        "Desired level of investment in business investment" @log qebfi
        "Desired level of consumption (FRBUS definition)" @log qec
        "Target level of consumption of durable goods, trending component" @log qecd
        "Desired level of consumption of nondurable goods and nonhousing services" @log qeco
        "Target level of residential investment" @log qeh
        "Desired level of civilian labor force" qlf
        "Trend labor force participation rate" qlfpr
        "Desired level of  business labor hours" qlhp
        "Trend workweek,  business sector (employee and  self-employed)" qlww
        "Desired level of consumption price" @log qpcnia
        "Desired level of compensation per hour, trending component" @log qpl
        "Desired price level of private output ex. energy, housing, and farm" @log qpxb
        "Desired level of nonconsumption price" @log qpxnc
        "Desired price level of adjusted final sales" @log qpxp
        "Desired level of dividends" @log qynidn
        "S&P BBB corporate bond rate" rbbb
        "After-tax real financial cost of capital for business investment" rbfi
        "Cost of capital for consumer durables" rccd
        "Cost of capital for residential investment" rcch
        "Real expected rate of return on equity" req
        "10-year Treasury bond rate" rg10
        "30-year Treasury bond rate" rg30
        "5-year Treasury note rate" rg5
        "Approximate average rate of interest on new federal debt" rgw
        "Real federal funds rate" rrff
        "Personal saving rate" rspnia
        "3-month Treasury bill rate" rtb
        "User cost of capital for business investment" rtbfi
        "User cost of capital for inventories" rtinv
        "Expected federal funds rate in the long run (Blue Chip)" rtr
        "Government corporate income tax accruals, current \$" @log tcin
        "Government personal income tax and non-tax receipts, current \$" @log tpn
        "Average tax rate on household income" tryh
        "Household property wealth ex. stock market, real" @log wpo
        "Household property wealth ex. stock market, current \$" @log wpon
        "Household stock market wealth, real" wps
        "Household stock market wealth, current \$" @log wpsn
        "Business output (BEA definition), cw 2012\$" @log xb
        "Business output (BEA definition), current \$" @log xbn
        "Business output, adjusted for measurement error, cw  2012\$" @log xbo
        "Potential business output, cw 2012\$" @log xbt
        "Final sales of gross domestic product, current \$" @log xfsn
        "Output gap for  business plus oil imports  (100*(actual/potential -1)" xgap
        "Output gap for GDP (100*(actual/potential -1)" xgap2
        "Gross domestic income, cw 2012\$" @log xgdi
        "Gross domestic income, current \$" @log xgdin
        "Gross domestic product, adjusted for measurement error, cw 2012\$" @log xgdo
        "GDP, current \$" @log xgdpn
        "Potential GDP, cw 2012\$" @log xgdpt
        "Potential GDP, current \$" @log xgdptn
        "Final sales plus imports less government labor, current \$" @log xpn
        "Disposable income" @log ydn
        "Income, household, total (real after-tax)" @log yh
        "Income, household, total, ratio to XGDP, cyclical component (real after-tax)" yhgap
        "Consumer interest payments to business" @log yhibn
        "Income, household, labor compensation (real after-tax)" @log yhl
        "Income, household, labor compensation" @log yhln
        "Income, household, property (real after-tax)" @log yhp
        "Income, household, property, ratio to YH, cyclical component (real after-tax)" yhpgap
        "Income, household, property, non-taxable component" yhpntn
        "Income, household, property, ratio to YH (real after-tax)" yhpshr
        "Income, household, property, taxable component" @log yhptn
        "Income, household, total, ratio to XGDP (real after-tax)" yhshr
        "Personal saving" @log yhsn
        "Income, household, transfer (real after-tax), net basis" @log yht
        "Income, household, transfer, ratio to YH, cyclical component (real after-tax)" yhtgap
        "Income, household, transfer payments. net basis" @log yhtn
        "Income, household, transfer, ratio to YH (real after-tax)" yhtshr
        "Income from stock of BFI" @log ykbfin
        "Income from stock of inventories" @log ykin
        "Corporate profits (national income component)" @log ynicpn
        "Labor income (national income component)" @log yniln
        "National income" @log ynin
        "Personal income" @log ypn
        "Expected growth rate of real dividends, for WPSN eq. (VAR exp.)" zdivgr
        "Expected growth rate of business output EBFI (VAR exp.)" zebfi
        "Expected growth rate of target durable consumption, for ECD eq. (VAR exp.)" zecd
        "Expected growth rate of target nondurables and nonhousing services, for ECO eq (VAR" zeco
        "Expected growth rate of target residential investment, for EH eq. (VAR exp.)" zeh
        "Expected output gap, for RG5 eq. (VAR exp.)" zgap05
        "Expected output gap, for RG10 eq. (VAR exp.)" zgap10
        "Expected output gap, for RG30 eq. (VAR exp.)" zgap30
        "Expected output gap, for ECD eq. (VAR exp.)" zgapc2
        "Expected growth rate of target aggregate hours (VAR exp.)" zlhp
        "Expected cons. price infl., for RCCH, RRMET, and YHPNTN eqs. (10-yr mat.) (VAR exp.)" zpi10
        "Expected cons. price infl., for FPXR eq. (10-yr mat.) (VAR exp.)" zpi10f
        "Expected cons. price infl., for RCCD eq. (5-yr mat.) (VAR exp.)" zpi5
        "Expected output price infl., for RPD eq. (5-yr mat.) (VAR exp.)" zpib5
        "Expected cons. price infl., for REQ eq. (30-yr mat.) (VAR exp.)" zpic30
        "Expected 4-qtr consumer price inflation (8 qtrs. in the future) (VAR exp.)" zpic58
        "Expected value of picxfe in the next quarter (VAR exp.)" zpicxfe
        "Expected value of pieci in the next quarter (VAR exp.)" zpieci
        "Expected federal funds rate, for RG10 eq. (10-yr mat.) (VAR exp.)" zrff10
        "Expected federal funds rate, for RG30 eq. (30-yr mat.) (VAR exp.)" zrff30
        "Expected federal funds rate, for RG5 eq. (5-yr mat.) (VAR exp.)" zrff5
        "Expected level of real after-tax household income, for QEC eq. (VAR exp.)" @log zyh
        "Expected level of real after-tax property income, for QEC eq. (VAR exp.)" @log zyhp
        "Expected level of real transfer income, for QEC eq. (VAR exp.)" @log zyht
        "Expected rate of growth of target real dividends, for YNIDN eq. (VAR exp.)" zynid

        # Behavioral variables:
        "Monetary policy indicator for unemployment threshold" dmptlur
        "Monetary policy indicator for both thresholds" dmptmax
        "Monetary policy indicator for inflation threshold" dmptpi
        "Monetary policy indicator for policy rule thresholds" dmptr
        "Price inflation aggregation adjustment" dpadj
        "Business Fixed Investment, cw 2012\$" @log ebfi
        "Consumption, cw 2012\$ (FRB/US definition)" @log ec
        "Consumer expenditures on durable goods, cw 2012\$" @log ecd
        "Consumer expenditures on housing services, cw 2012\$" @log ech
        "Personal consumption expenditures, cw 2012\$ (NIPA definition)" @log ecnia
        "Consumer expenditures on non-durable goods and non-housing services, cw 2012\$" @log eco
        "Federal Government expenditures, CW 2012\$" @log egfe
        "Federal Government expenditures, CW 2012\$, Trend" @log egfet
        "Federal government employee compensation, cw 2012\$" @log egfl
        "Federal government employee compensation, cw 2012\$, trend" @log egflt
        "S&L Government expenditures, CW 2012\$" @log egse
        "S&L Government expenditures, CW 2012\$, Trend" @log egset
        "S&L government employee compensation, cw 2012\$" @log egsl
        "S&L government employee compensation, cw 2012\$, trend" @log egslt
        "Residential investment expenditures, cw 2012\$" @log eh
        "Change in business inventories, current \$" ein
        "Imports of goods and services ex. petroleum, cw 2012\$" @log emo
        "Petroleum imports, cw 2012\$" @log emp
        "Exports of goods and services, cw 2012\$" @log ex
        "Foreign aggregate GDP (world, bilateral export weights), trend" @log fgdpt
        "Foreign consumer price inflation (G10)" fpi10
        "Foreign consumer price inflation, trend (G10)" fpi10t
        "Foreign consumer price inflation (G39, bilateral export trade weights)" fpic
        "Real exchange rate (G39, import/export trade weights)" fpxr
        "Real exchange rate residual" fpxrr
        "Foreign long-term interest rate (G10)" frl10
        "Foreign short-term interest rate (G10)" frs10
        "Equilibrium real short-term interest rate used in foreign Taylor rule" frstar
        "Foreign output gap (world, bilateral export weights)" fxgap
        "Federal government net transfer payments, current \$" @log gtn
        "Government net transfer payments, deflated by PGDP" @log gtr
        "Deviation of ratio of government transfers to GDP from trend ratio" gtrd
        "Petroleum imports, cw 2012\$, trend growth rate" hgemp
        "Trend growth rate of PKIR" hgpkir
        "Growth rate of KS, cw 2012\$ (compound annual rate)" hks
        "Trend growth rate of LEP (annual rate)" hlept
        "Trend growth rate of multifactor productivity" hmfpt
        "Drift component of change in QLFPR" hqlfpr
        "Trend growth rate of workweek" hqlww
        "Drift term in stochastic component of trend ratio of PCNIA to PXP" huqpct
        "Drift term in UXBT" huxb
        "Residual Factor (Trend rate of growth of XB)" hxbtr
        "Consumption of fixed capital, current \$" @log jccan
        "Stock of private inventories, cw 2012\$" @log ki
        "Difference between household and business sector payroll employment, less government employment" leo
        "Labor force participation rate" lfpr
        "Aggregate labor hours,  business sector (employee and  self-employed)" lhp
        "Civilian unemployment rate (break adjusted)" lur
        "Natural rate of unemployment" lurnat
        "Workweek,  business sector (employee and self-employed)" lww
        "Multiplicative discrepancy for the difference between XGDI and XGDO" mei
        "Multiplicative discrepancy for the difference between XGDP and XGDO" mep
        "Multifactor productivity, trend level" mfpt
        "Price level of BFI compared to PXP" pbfir
        "Price index for consumer durables, cw (relative to to PCNIA)" pcdr
        "Price index for personal consumption expenditures on energy (relative to PCXFE)" pcer
        "Price index for personal consumption expenditures on food (relative to PCXFE)" pcfr
        "Price index for housing services, cw (relative to to PCNIA)" pchr
        "Price index for non-durable goods and non-housing services, cw (relative to to PCNIA" pcor
        "Price index for personal consumption expendits ex. food and energy, cw (NIPA definit" @log pcxfe
        "Price index for federal government expenditures, CW (relative to PXP)" pegfr
        "Price index for S&L government expenditures, CW (relative to PXP)" pegsr
        "Loan Performance House Price Index" @log phouse
        "Price index for residential investment, cw (relative to PXP)" phr
        "Inflation rate, personal consumption expenditures, cw" picnia
        "Inflation rate, personal consumption expenditures, ex. food and energy, cw" picxfe
        "Annualized rate of growth of EI hourly compensation" pieci
        "Rate of growth of PL" pipl
        "Inflation rate, price of adjusted final sales excluding consumption (annual rate)" pipxnc
        "Price index for imports ex. petroleum, cw" @log pmo
        "Price of imported oil, relative to price index for bus. sector output" poilr
        "10-year expected PCE price inflation (Survey of Professional Forecasters)" ptr
        "Price of adjusted final sales excluding consumption" @log pxnc
        "Price index for final sales plus imports less gov. labor" @log pxp
        "Price index for exports, cw (relative to PXP)" pxr
        "Desired Inventory Sales Ratio" qkir
        "Random walk component of non-oil import prices" qpmo
        "S&P BBB corporate bond rate, risk/term premium" rbbbp
        "New car loan rate at finance companies" rcar
        "Rate of capital gain on the non-equity portion of household wealth" rcgain
        "Real expected rate of return on equity, premium component" reqp
        "Federal funds rate" rff
        "Value of eff. federal funds rate given by estimated policy rule" rffalt
        "Federal funds rate" rffrule
        "Value of eff. federal funds rate given by the inertial Taylor rule" rffintay
        "Value of eff. federal funds rate given by the Taylor rule with output gap" rfftay
        "Value of eff. federal funds rate given by the Taylor rule with unemployment gap" rfftlr
        "Average yield earned on gross claims of US residents on the rest of the world" rfynic
        "Average yield earned on liabilities of US residents on the rest of the world" rfynil
        "10-year Treasury bond rate, term premium" rg10p
        "30-year Treasury bond rate, term premium" rg30p
        "5-year Treasury note rate. term premium" rg5p
        "Average rate of interest on existing federal debt" rgfint
        "Interest rate on conventional mortgages" rme
        "Expected long-run real federal funds rate" rrtr
        "Equilibrium real federal funds rate (for monetary policy reaction functions)" rstar
        "Average government corporate income tax rate" trci
        "Average government tax rate for personal income tax and non-tax receipts" trp
        "Average government tax rate for personal income tax, trend" trpt
        "Federal Government budget surplus, residual" ugfsrp
        "Multiplicative factor for government civilian employment" uleg
        "Stochastic component of trend ratio of PCNIA to PXP" uqpct
        "Stochastic component of trend ratio of XGDPT to XBT" uxbt
        "Corporate profits, residual" uynicpnr
        "Desired investment-output ratio" vbfi
        "Residual Factor (Potential business output)" xbtr
        "Final sales of gross domestic product, cw 2012\$" @log xfs
        "GDP, cw 2012\$" @log xgdp
        "Final sales plus imports less government labor, cw 2012\$" @log xp
        "Imputed income of the stock of consumer durables, 2012\$" @log yhpcd
        "Dividends (national income component)" @log ynidn
        "Net interest, rental and proprietors' incomes (national income components)" @log ynirn
        "Expected trend share of property income in household income" zyhpst
        "Expected trend ratio of household income to GDP" zyhst
        "Expected trend share of transfer income in household income" zyhtst
    end

    @exogenous m begin
        # Exogenous variables:
        "Potential government employment ratio (relative to business)" adjlegrt
        "Dummy, post-1979 indicator" d79a
        "Dummy, 1980-1995 indicator" d8095
        "Dummy, post-1983 indicator" d83
        "Dummy, post-1986 indicator" d87
        "Dock strike dummy, import equation" ddockm
        "Dock strike dummy, export equation" ddockx
        "EUC switch:  1 for including EUC, 0 for not including" deuc
        "Dummy, Foreign monetary policy switch:  Exogenous real interest rate" dfmprr
        "Fiscal policy switch:  1 for debt ratio stabilization" dfpdbt
        "Fiscal policy switch:  1 for exogenous personal income trend tax rates" dfpex
        "Fiscal policy switch:  1 for surplus ratio stabilization" dfpsrp
        "Switch to control for long-run productivity growth in the government sector" dglprd
        "Monetary policy switch: MA rule" dmpalt
        "Monetary policy switch:  exogenous federal funds rate" dmpex
        "Monetary policy switch:  Generalized reaction function" dmpgen
        "Monetary policy switch:  inertial taylor rule" dmpintay
        "Monetary policy switch:  exogenous real federal funds rate" dmprr
        "Monetary policy switch:  Taylor's reaction function" dmptay
        "Monetary policy switch:  Taylor's reaction function with unemployment gap" dmptlr
        "Monetary policy threshold switch:  0 for no threshold,  1 for threshold" dmptrsh
        "RSTAR updating switch: 1 is on, 0 is off" drstar
        "Oil imports to GDP ratio (Trend)" emptrt
        "Foreign target consumer price inflation (G10)" fpitrg
        "Real exchange rate residual, trend" fpxrrt
        "Federal government target debt-to-GDP ratio" gfdrt
        "Federal government target surplus-to-GDP ratio" gfsrt
        "Government, trend ratio of transfer payments to GDP" gtrt
        "Trend growth rate of price of consumer durable goods (relative to PCNIA)" hgpcdr
        "Residual growth of capital services" hksr
        "Depreciation rate, business investment" jrbfi
        "Depreciation rate, consumer durables" jrcd
        "Depreciation rate, housing" jrh
        "Emergency unemployment compensation (EUC)" leuc
        "Labor quality, trend level" lqualt
        "Unemployment threshold" lurtrsh
        "Noninstitutional population, aged 16 and over (break adjusted)" n16
        "Real PCE price of food, trend" pcfrt
        "Target consumption price level (used in RFFGEN policy rule)" pcstar
        "Target rate of consumption price inflation (used in policy reaction functions)" pitarg
        "Inflation threshold" pitrsh
        "Price index for stock of inventories, cw (relative to PXP)" pkir
        "Price of imported oil, relative to price index for bus. sector output, trend" poilrt
        "Equilibrium business price markup" pwstar
        "Desired ratio of employment discrepancy to the labor force" qleor
        "Federal funds rate given by fixed, pre-determined funds rate path" rfffix
        "Minimum nominal funds rate (set at 0 to impose zero lower bound)" rffmin
        "Residual in FNICN equation" rfnict
        "Real foreign short-term interest rate" rfrs10
        "Real federal funds rate given by fixed, pre-determined real funds rate path" rrfix
        "Time trend, begins in 1947q1 (0 before)" t47
        "Proportion of investment tax credit deducted from depr. base" tapddp
        "Present value of depreciation allowances, BFI" tdpv
        "Government corporate income tax rate, trend" trcit
        "Marginal federal corporate income tax rate" trfcim
        "Marginal federal personal income tax rate (at twice median family income)" trfpm
        "Investment tax credit for business investment" tritc
        "Average tax rate for personal income tax, trend, policy setting" trptx
        "Marginal S&L tax rate on personal property" trspp
        "Trend in ratio of EMON to XGDEN" uemot
        "Multiplicative factor in FCBRN identity" ufcbr
        "Multiplicative factor in FNIRN identity" ufnir
        "Multiplicative factor in FTCIN identity" uftcin
        "Multiplicative factor in GFDBTN identity" ugfdbt
        "Federal government debt ratio" ugfdbtp
        "Multiplicative factor in PCPI identity" upcpi
        "Multiplicative factor in PCPIX identity" upcpix
        "Multiplicative factor in PGFL identity" upgfl
        "Multiplicative factor in PGSL identity" upgsl
        "Multiplicative factor in PKBFIR identity" upkbfir
        "Multiplicative factor in PMP identity" upmp
        "Multiplicative factor in PXB   identity" upxb
        "Multiplicative factor in VBFI identity" uvbfi
        "Multiplicative factor in YDN identity" uyd
        "Multiplicative factor (Consumer interest payments to business)" uyhibn
        "Multiplicative factor in YHLN identity" uyhln
        "Multiplicative factor in YHPTN identity" uyhptn
        "Multiplicative factor in personal saving identity (accounts for transfers to foreign" uyhsn
        "Multiplicative factor in YHTN identity" uyhtn
        "Multiplicative factor in YLN identity" uyl
        "Multiplicative factor in YNIN identity" uyni
        "Multiplicative factor in YPN identity" uyp
        "Microsoft one-time dividend payout in 2004Q4" ymsdn
    end

    @autoshocks m _a

    @equations m begin
        # Identity equations:
        "Federal funds rate, first diff"
        delrff[t] - delrff_a[t] = rff[t] - rff[t - 1]
        "Price inflation aggregation discrepancy"
        dpgap[t] - dpgap_a[t] = pipxnc[t] / 400 - (y_dpgap[1] * @d(log(phr[t] * pxp[t]), 0, 1) + y_dpgap[2] * @d(log(pbfir[t] * pxp[t]), 0, 1) + (y_dpgap[3] + y_dpgap[4]) * @d(log(pegfr[t] * pxp[t]), 0, 1) + (y_dpgap[5] + y_dpgap[6]) * @d(log(pegsr[t] * pxp[t]), 0, 1) + y_dpgap[7] * @d(log(pxr[t] * pxp[t]), 0, 1))
        "Investment in equipment, current \$"
        ebfin[t] - ebfin_a[t] = 0.01 * ebfi[t] * pbfir[t] * pxp[t]
        "Personal consumption expenditures, current \$ (NIPA definition)"
        ecnian[t] - ecnian_a[t] = 0.01 * pcnia[t] * ecnia[t]
        "Federal Government expenditures, current \$"
        egfen[t] - egfen_a[t] = pegfr[t] * pxp[t] * egfe[t] * 0.01
        "Federal government employee compensation, current \$"
        egfln[t] - egfln_a[t] = 0.01 * pgfl[t] * egfl[t]
        "S&L Government expenditures, current \$"
        egsen[t] - egsen_a[t] = pegsr[t] * pxp[t] * egse[t] * 0.01
        "S&L government employee compensation, current \$"
        egsln[t] - egsln_a[t] = 0.01 * pgsl[t] * egsl[t]
        "Residential investment expenditures"
        ehn[t] - ehn_a[t] = 0.01 * phr[t] * pxp[t] * eh[t]
        "Change in private inventories, cw 2012\$"
        ei[t] - ei_a[t] = 4 * @d(ki[t], 0, 1)
        "Imports of goods and services, cw 2012\$"
        log(em[t]) - em_a[t] = log(em[t - 1]) + 0.5 * (emon[t] / emn[t] + emon[t - 1] / emn[t - 1]) * @d(log(emo[t]), 0, 1) + 0.5 * (empn[t] / emn[t] + empn[t - 1] / emn[t - 1]) * @d(log(emp[t]), 0, 1)
        "Imports of goods and services, current \$"
        emn[t] - emn_a[t] = emon[t] + empn[t]
        "Imports of goods and services ex. petroleum"
        emon[t] - emon_a[t] = 0.01 * pmo[t] * emo[t]
        "Petroleum imports, current \$"
        empn[t] - empn_a[t] = 0.01 * pmp[t] * emp[t]
        "Exports of goods and services, current \$"
        exn[t] - exn_a[t] = 0.01 * pxp[t] * pxr[t] * ex[t]
        "US current account balance, current \$"
        fcbn[t] - fcbn_a[t] = (exn[t] - emn[t]) + fynin[t] + fcbrn[t]
        "US current account balance residual, current \$"
        fcbrn[t] - fcbrn_a[t] = (ufcbr[t] * pxb[t] * xbt[t]) / 100
        "Foreign aggregate GDP (world, bilateral export weights)"
        fgdp[t] - fgdp_a[t] = fgdpt[t] * exp(fxgap[t] / 100)
        "Gross stock of claims of US residents on the rest of the world, current \$"
        @d(fnicn[t], 0, 1) / xgdptn[t] - fnicn_a[t] = ((0.54 * @d(log(fpc[t]), 0, 1) * fnicn[t - 1]) / xgdptn[t] - (0.67 * @d(log(fpx[t]), 0, 1) * fnicn[t - 1]) / xgdptn[t]) + rfnict[t]
        "Gross stock of liabilities of US residents to the rest of the world, current \$"
        fniln[t] - fniln_a[t] = fnicn[t] - fnin[t]
        "Net stock of claims of US residents on the rest of the world, current \$"
        @d(fnin[t], 0, 1) - fnin_a[t] = (((0.25 * fcbn[t] + 0.54 * (@d(log(fpc[t]), 0, 1) * fnicn[t - 1])) - 0.32 * (@d(log(pgdp[t]), 0, 1) * fniln[t - 1])) - 0.67 * (@d(log(fpx[t]), 0, 1) * fnicn[t - 1])) + 0.06 * (@d(log(fpx[t]), 0, 1) * fniln[t - 1]) + fnirn[t]
        "Net stock of claims of US residents on the rest of the world, residual"
        fnirn[t] - fnirn_a[t] = ufnir[t] * xgdpn[t]
        "Foreign aggregate consumer price (G39, import/export trade weights)"
        fpc[t] - fpc_a[t] = fpc[t - 1] * exp(fpic[t] / 400)
        "Nominal exchange rate (G39, import/export trade weights)"
        fpx[t] - fpx_a[t] = (fpxr[t] * fpc[t]) / pcpi[t]
        "Corporate taxes paid to rest of world, current \$"
        ftcin[t] - ftcin_a[t] = uftcin[t] * ynicpn[t]
        "Gross investment income received from the rest of the world, current \$"
        fynicn[t] - fynicn_a[t] = 0.01 * rfynic[t] * fnicn[t - 1]
        "Gross investment income paid to the rest of the world, current \$"
        fyniln[t] - fyniln_a[t] = 0.01 * rfynil[t] * fniln[t - 1]
        "Net investment income received from the rest of the world, current \$"
        fynin[t] - fynin_a[t] = fynicn[t] - fyniln[t]
        "Federal government debt stock, current \$"
        gfdbtn[t] - gfdbtn_a[t] = ugfdbt[t] * gfdbtnp[t]
        "Federal government debt stock held by the public, current \$"
        gfdbtnp[t] - gfdbtnp_a[t] = ugfdbtp[t] * (gfdbtnp[t - 1] - 0.25 * gfsrpn[t])
        "Government payments"
        gfexpn[t] - gfexpn_a[t] = egfln[t] + egfen[t] + gtn[t] + gfintn[t]
        "Federal government net interest payments, current \$"
        gfintn[t] - gfintn_a[t] = rgfint[t] * gfdbtn[t - 1]
        "Government receipts and residual"
        gfrecn[t] - gfrecn_a[t] = tpn[t] + tcin[t] + ugfsrp[t] * xgdpn[t]
        "Federal government budget surplus, current \$"
        gfsrpn[t] - gfsrpn_a[t] = (((((tpn[t] + tcin[t]) - egfln[t]) - egfen[t]) - gtn[t]) - gfintn[t]) + ugfsrp[t] * xgdpn[t]
        "Growth rate of GDP, cw 2012\$ (annual rate)"
        hggdp[t] - hggdp_a[t] = 400 * @d(log(xgdp[t]), 0, 1)
        "Trend growth rate of XGDP, cw 2012\$ (annual rate)"
        hggdpt[t] - hggdpt_a[t] = hxbt[t] + huxb[t]
        "Trend growth rate of price of business investment (relative to PXB)"
        hgpbfir[t] - hgpbfir_a[t] = 0.975 * hgpbfir[t - 1] + 0.025 * 400 * log(((pbfir[t] * pxp[t]) / pxb[t]) / ((pbfir[t - 1] * pxp[t - 1]) / pxb[t - 1]))
        "Growth rate of real after-tax corporate profits"
        hgynid[t] - hgynid_a[t] = 400 * @d(log(((ynicpn[t] - tcin[t]) * 0.5) / pxb[t]), 0, 1)
        "Trend growth rate of output per hour"
        hlprdt[t] - hlprdt_a[t] = (hxbt[t] - hlept[t]) - hqlww[t]
        "Trend rate of growth of XB  , cw 2012\$ (annual rate)"
        hxbt[t] - hxbt_a[t] = 0.725 * (hlept[t] + hqlww[t] + 400 * @d(log(lqualt[t]), 0, 1)) + 0.275 * hks[t] + hmfpt[t] + hxbtr[t]
        "Consumption of fixed capital, consumer durables"
        jkcd[t] - jkcd_a[t] = jrcd[t] * kcd[t - 1]
        "Capital stock - BFI, 2012\$"
        kbfi[t] - kbfi_a[t] = ((pbfir[t] / pkbfir[t]) * ebfi[t]) / 4 + (1 - jrbfi[t] / 4) * kbfi[t - 1]
        "Stock of consumer durables, cw 2012\$"
        kcd[t] - kcd_a[t] = 0.25 * ecd[t] + (1 - jrcd[t] / 4) * kcd[t - 1]
        "Stock of residential structures, cw 2012\$"
        kh[t] - kh_a[t] = 0.25 * eh[t] + (1 - jrh[t] / 4) * kh[t - 1]
        "Capital services, 2012\$"
        log(ks[t]) - ks_a[t] = log(ks[t - 1]) + hks[t] / 400
        "Government civilian employment ex. gov. enterprise"
        log(leg[t]) - leg_a[t] = (log(uleg[t]) + log(egfl[t] + egsl[t])) - log(lprdt[t])
        "Civilian employment (break adjusted)"
        leh[t] - leh_a[t] = lep[t] + leg[t] + leo[t]
        "Employment in  business sector (employee and  self-employed)"
        lep[t] - lep_a[t] = lhp[t] / lww[t]
        "Potential employment in  business sector"
        leppot[t] - leppot_a[t] = (qlf[t] * ((1 - 0.01 * lurnat[t]) - qleor[t])) / (1 + adjlegrt[t])
        "Civilian labor force (break adjusted)"
        lf[t] - lf_a[t] = lfpr[t] * n16[t]
        "Trend labor productivity"
        log(lprdt[t]) - lprdt_a[t] = (log(xbt[t]) - log(leppot[t])) - log(qlww[t])
        "Price index for personal consumption expenditures, cw (NIPA definition)"
        @d(log(pcnia[t]), 0, 1) - pcnia_a[t] = picnia[t] / 400
        "Consumer price index,total"
        pcpi[t] - pcpi_a[t] = upcpi[t] * pcnia[t]
        "Consumer price index,excluding food and energy"
        pcpix[t] - pcpix_a[t] = upcpix[t] * pcxfe[t]
        "Price index for GDP, cw"
        pgdp[t] - pgdp_a[t] = (100 * xgdpn[t]) / xgdp[t]
        "Price index for federal government employee compensation, cw"
        log(pgfl[t]) - pgfl_a[t] = (log(upgfl[t]) + log(pl[t])) - log(lprdt[t])
        "Price index for S&L government employee compensation, cw"
        log(pgsl[t]) - pgsl_a[t] = (log(upgsl[t]) + log(pl[t])) - log(lprdt[t])
        "Four-quarter percent change in PCE prices"
        pic4[t] - pic4_a[t] = 100 * (pcnia[t] / pcnia[t - 4] - 1)
        "Four-quarter percent change core in PCE prices"
        picx4[t] - picx4_a[t] = 100 * (pcxfe[t] / pcxfe[t - 4] - 1)
        "Inflation rate, GDP, cw"
        pigdp[t] - pigdp_a[t] = 400 * @d(log(pgdp[t]), 0, 1)
        "Ratio of price of BFI stock (KBFI) to PXP"
        pkbfir[t] - pkbfir_a[t] = upkbfir[t] * pbfir[t]
        "Compensation per hour,  business"
        log(pl[t]) - pl_a[t] = log(pl[t - 1]) + pipl[t] / 400
        "Price index for petroleum imports"
        pmp[t] - pmp_a[t] = upmp[t] * poil[t]
        "Price of imported oil (\$ per barrel)"
        poil[t] - poil_a[t] = poilr[t] * pxb[t]
        "Price index for business output"
        pxb[t] - pxb_a[t] = upxb[t] * pgdp[t]
        "Desired level of investment in business investment"
        qebfi[t] - qebfi_a[t] = xb[t] * vbfi[t] * ((hxbt[t] / 100 - 0.01 * hgpbfir[t]) + jrbfi[t])
        "Desired level of consumption (FRBUS definition)"
        log(qec[t]) - qec_a[t] = y_qec[1] + y_qec[2] * log((zyh[t] - zyht[t]) - zyhp[t]) + y_qec[3] * log(zyht[t]) + (((1 - y_qec[2]) - y_qec[3]) - y_qec[4]) * log(zyhp[t]) + y_qec[4] * log(wpo[t] + wps[t])
        "Target level of consumption of durable goods, trending component"
        qecd[t] - qecd_a[t] = qec[t] * (jrcd[t] / 4 + ((hggdpt[t] + hggdpt[t - 1] + hggdpt[t - 2] + hggdpt[t - 3] + hggdpt[t - 4] + hggdpt[t - 5] + hggdpt[t - 6] + hggdpt[t - 7]) / 8) / 400 + (y_qecd[1] * hgpcdr[t]) / 400) * exp(y_qecd[2] + y_qecd[3] * (log(pcdr[t]) + log(rccd[t])))
        "Desired level of consumption of nondurable goods and nonhousing services"
        log(qeco[t]) - qeco_a[t] = (y_qeco[1] + log(qec[t])) - log(pcor[t])
        "Target level of residential investment"
        qeh[t] - qeh_a[t] = qec[t] * (jrh[t] / 4 + ((hggdpt[t] + hggdpt[t - 1] + hggdpt[t - 2] + hggdpt[t - 3] + hggdpt[t - 4] + hggdpt[t - 5] + hggdpt[t - 6] + hggdpt[t - 7] + hggdpt[t - 8] + hggdpt[t - 9] + hggdpt[t - 10] + hggdpt[t - 11] + hggdpt[t - 12] + hggdpt[t - 13] + hggdpt[t - 14] + hggdpt[t - 15]) / 16) / 400) * exp((y_qeh[1] - log((phr[t] * pxp[t]) / pcnia[t])) + y_qeh[2] * log(rcch[t]))
        "Desired level of civilian labor force"
        qlf[t] - qlf_a[t] = qlfpr[t] * n16[t]
        "Trend labor force participation rate"
        qlfpr[t] - qlfpr_a[t] = qlfpr[t - 1] + hqlfpr[t]
        "Desired level of  business labor hours"
        qlhp[t] - qlhp_a[t] = xbo[t] / lprdt[t]
        "Trend workweek,  business sector (employee and  self-employed)"
        log(qlww[t]) - qlww_a[t] = log(qlww[t - 1]) + hqlww[t - 1] / 400
        "Desired level of consumption price"
        log(qpcnia[t]) - qpcnia_a[t] = log(qpxp[t]) + log(uqpct[t])
        "Desired level of compensation per hour, trending component"
        log(qpl[t]) - qpl_a[t] = log(pl[t]) + y_qpl[1] * log(pxb[t] / qpxb[t])
        "Desired price level of private output ex. energy, housing, and farm"
        log(qpxb[t]) - qpxb_a[t] = log(pwstar[t]) + y_qpxb[1] + y_qpxb[2] * log(pl[t] / lprdt[t])
        "Desired level of nonconsumption price"
        log(qpxnc[t]) - qpxnc_a[t] = log(pxnc[t]) + y_qpxnc[1] * log(qpxp[t] / pxp[t]) + y_qpxnc[2] * log(qpcnia[t] / pcnia[t])
        "Desired price level of adjusted final sales"
        qpxp[t] - qpxp_a[t] = (100 * (xpn[t] + (0.01 * qpxb[t] * xb[t] - xbn[t]))) / xp[t]
        "Desired level of dividends"
        log(qynidn[t]) - qynidn_a[t] = y_qynidn[1] + y_qynidn[2] * d79a[t] + y_qynidn[3] * log(max(ynicpn[t] - tcin[t], 0.01))
        "S&P BBB corporate bond rate"
        rbbb[t] - rbbb_a[t] = rbbbp[t] + rg10[t]
        "After-tax real financial cost of capital for business investment"
        rbfi[t] - rbfi_a[t] = 0.5 * ((7.2 + (1 - trfcim[t]) * ((rg5[t] + rbbb[t]) - rg10[t])) - zpib5[t]) + 0.5 * req[t]
        "Cost of capital for consumer durables"
        rccd[t] - rccd_a[t] = max((100 * jrcd[t] + rcar[t]) - zpi5[t], 0.01)
        "Cost of capital for residential investment"
        rcch[t] - rcch_a[t] = max((100 * jrh[t] + (1 - trfpm[t] / 100) * (rme[t] + 100 * trspp[t])) - zpi10[t], 0.1)
        "Real expected rate of return on equity"
        req[t] - req_a[t] = (rg30[t] - zpic30[t]) + reqp[t]
        "10-year Treasury bond rate"
        rg10[t] - rg10_a[t] = zrff10[t] + rg10p[t]
        "30-year Treasury bond rate"
        rg30[t] - rg30_a[t] = zrff30[t] + rg30p[t]
        "5-year Treasury note rate"
        rg5[t] - rg5_a[t] = zrff5[t] + rg5p[t]
        "Approximate average rate of interest on new federal debt"
        rgw[t] - rgw_a[t] = y_rgw[1] * rtb[t] + y_rgw[2] * rg5[t] + y_rgw[3] * rg10[t] + y_rgw[4] * rg30[t]
        "Real federal funds rate"
        rrff[t] - rrff_a[t] = rff[t] - (picxfe[t] + picxfe[t - 1] + picxfe[t - 2] + picxfe[t - 3]) / 4
        "Personal saving rate"
        rspnia[t] - rspnia_a[t] = (100 * yhsn[t]) / ydn[t]
        "3-month Treasury bill rate"
        rtb[t] - rtb_a[t] = y_rtb[1] + y_rtb[2] * rtb[t - 1] + y_rtb[3] * rtb[t - 2] + y_rtb[4] * rff[t] + y_rtb[5] * rff[t - 1]
        "User cost of capital for business investment"
        rtbfi[t] - rtbfi_a[t] = ((0.01 * rbfi[t] + jrbfi[t]) - 0.01 * hgpbfir[t]) * (((1 - 0.01 * tritc[t]) - trfcim[t] * (1 - tapddp[t] * 0.01 * tritc[t]) * tdpv[t]) / (1 - trfcim[t])) * ((pkbfir[t] * pxp[t]) / pxb[t])
        "User cost of capital for inventories"
        rtinv[t] - rtinv_a[t] = ((0.01 * rbfi[t] - 0.01 * hgpkir[t]) * ((pxp[t] * pkir[t] + pxp[t - 1] * pkir[t - 1]) / 2)) / pxb[t]
        "Expected federal funds rate in the long run (Blue Chip)"
        rtr[t] - rtr_a[t] = rrtr[t] + ptr[t]
        "Government corporate income tax accruals, current \$"
        tcin[t] - tcin_a[t] = trci[t] * ynicpn[t]
        "Government personal income tax and non-tax receipts, current \$"
        tpn[t] - tpn_a[t] = trp[t] * (ypn[t] - gtn[t])
        "Average tax rate on household income"
        tryh[t] - tryh_a[t] = tpn[t] / (yhln[t] + yhptn[t])
        "Household property wealth ex. stock market, real"
        wpo[t] - wpo_a[t] = wpon[t] / (0.01 * pcnia[t])
        "Household property wealth ex. stock market, current \$"
        wpon[t] - wpon_a[t] = wpon[t - 1] * exp(((1 - y_wpon[1]) * rcgain[t]) / 400 + y_wpon[1] * (log(phouse[t]) - log(phouse[t - 1]))) + 0.25 * ((ydn[t] - ecnian[t]) - yhibn[t]) + 0.25 * (0.01 * pcdr[t] * pcnia[t] * (ecd[t] - jkcd[t]))
        "Household stock market wealth, real"
        wps[t] - wps_a[t] = wpsn[t] / (0.01 * pcnia[t])
        "Household stock market wealth, current \$"
        log(wpsn[t]) - wpsn_a[t] = (log((ynicpn[t] - tcin[t]) * 0.5) - 0.25 * (req[t] - zdivgr[t])) + log(25) + 1
        "Business output (BEA definition), cw 2012\$"
        xb[t] - xb_a[t] = xbn[t] / (pxb[t] / 100)
        "Business output (BEA definition), current \$"
        xbn[t] - xbn_a[t] = ((pxb[t] / 100) * xbo[t] + xgdpn[t]) - (xgdo[t] * pgdp[t]) / 100
        "Business output, adjusted for measurement error, cw  2012\$"
        log(xbo[t]) - xbo_a[t] = log(xbt[t]) + (y_xbo[1] * xgap2[t]) / 100
        "Potential business output, cw 2012\$"
        log(xbt[t]) - xbt_a[t] = y_xbt[1] * (log(leppot[t]) + log(qlww[t]) + log(lqualt[t])) + y_xbt[2] * log(ks[t]) + log(mfpt[t]) + log(xbtr[t])
        "Final sales of gross domestic product, current \$"
        xfsn[t] - xfsn_a[t] = xgdpn[t] - ein[t]
        "Output gap for  business plus oil imports  (100*(actual/potential -1)"
        xgap[t] - xgap_a[t] = 100 * (xbo[t] / xbt[t] - 1)
        "Output gap for GDP (100*(actual/potential -1)"
        xgap2[t] - xgap2_a[t] = 100 * (xgdo[t] / xgdpt[t] - 1)
        "Gross domestic income, cw 2012\$"
        xgdi[t] - xgdi_a[t] = xgdo[t] * mei[t]
        "Gross domestic income, current \$"
        xgdin[t] - xgdin_a[t] = xgdi[t] * (pgdp[t] / 100)
        "Gross domestic product, adjusted for measurement error, cw 2012\$"
        xgdo[t] - xgdo_a[t] = xgdp[t] / mep[t]
        "GDP, current \$"
        xgdpn[t] - xgdpn_a[t] = ((xpn[t] + ein[t]) - emn[t]) + egfln[t] + egsln[t]
        "Potential GDP, cw 2012\$"
        log(xgdpt[t]) - xgdpt_a[t] = log(xbt[t]) + log(uxbt[t])
        "Potential GDP, current \$"
        xgdptn[t] - xgdptn_a[t] = 0.01 * pgdp[t] * xgdpt[t]
        "Final sales plus imports less government labor, current \$"
        xpn[t] - xpn_a[t] = 0.01 * pxp[t] * xp[t]
        "Disposable income"
        ydn[t] - ydn_a[t] = uyd[t] * (ypn[t] - tpn[t])
        "Income, household, total (real after-tax)"
        yh[t] - yh_a[t] = yhl[t] + yht[t] + yhp[t]
        "Income, household, total, ratio to XGDP, cyclical component (real after-tax)"
        yhgap[t] - yhgap_a[t] = 100 * log(yhshr[t] / zyhst[t])
        "Consumer interest payments to business"
        yhibn[t] - yhibn_a[t] = uyhibn[t] * xgdpn[t]
        "Income, household, labor compensation (real after-tax)"
        yhl[t] - yhl_a[t] = ((1 - tryh[t]) * yhln[t]) / (0.01 * pcnia[t])
        "Income, household, labor compensation"
        yhln[t] - yhln_a[t] = uyhln[t] * yniln[t]
        "Income, household, property (real after-tax)"
        yhp[t] - yhp_a[t] = ((1 - tryh[t]) * yhptn[t] + yhpntn[t]) / (0.01 * pcnia[t])
        "Income, household, property, ratio to YH, cyclical component (real after-tax)"
        yhpgap[t] - yhpgap_a[t] = 100 * log(yhpshr[t] / zyhpst[t])
        "Income, household, property, non-taxable component"
        yhpntn[t] - yhpntn_a[t] = ((((0.01 * pcnia[t] * pcdr[t] * yhpcd[t] - yhibn[t]) + ynicpn[t]) - tcin[t]) - ynidn[t]) - 0.01 * zpi10[t] * gfdbtn[t]
        "Income, household, property, ratio to YH (real after-tax)"
        yhpshr[t] - yhpshr_a[t] = yhp[t] / yh[t]
        "Income, household, property, taxable component"
        yhptn[t] - yhptn_a[t] = uyhptn[t] * (ynirn[t] + gfintn[t] + ynidn[t] + yhibn[t])
        "Income, household, total, ratio to XGDP (real after-tax)"
        yhshr[t] - yhshr_a[t] = yh[t] / xgdp[t]
        "Personal saving"
        yhsn[t] - yhsn_a[t] = ((((yhln[t] + yhtn[t] + yhptn[t]) - tpn[t]) - ecnian[t]) - yhibn[t]) + uyhsn[t] * xgdptn[t]
        "Income, household, transfer (real after-tax), net basis"
        yht[t] - yht_a[t] = yhtn[t] / (0.01 * pcnia[t])
        "Income, household, transfer, ratio to YH, cyclical component (real after-tax)"
        yhtgap[t] - yhtgap_a[t] = 100 * log(yhtshr[t] / zyhtst[t])
        "Income, household, transfer payments. net basis"
        yhtn[t] - yhtn_a[t] = uyhtn[t] * gtn[t]
        "Income, household, transfer, ratio to YH (real after-tax)"
        yhtshr[t] - yhtshr_a[t] = yht[t] / yh[t]
        "Income from stock of BFI"
        ykbfin[t] - ykbfin_a[t] = (0.01 * rtbfi[t] * pxb[t] * (kbfi[t] + kbfi[t - 1])) / 2
        "Income from stock of inventories"
        ykin[t] - ykin_a[t] = (0.01 * rtinv[t] * pxb[t] * (ki[t] + ki[t - 1])) / 2
        "Corporate profits (national income component)"
        ynicpn[t] - ynicpn_a[t] = max(((ynin[t] - yniln[t]) - ynirn[t]) + uynicpnr[t] * xgdpn[t], tcin[t] + 0.01 * xgdpn[t])
        "Labor income (national income component)"
        yniln[t] - yniln_a[t] = 0.01 * uyl[t] * (pl[t] * lhp[t] + pgfl[t] * egfl[t] + pgsl[t] * egsl[t])
        "National income"
        ynin[t] - ynin_a[t] = uyni[t] * ((xgdin[t] + fynin[t]) - jccan[t])
        "Personal income"
        ypn[t] - ypn_a[t] = uyp[t] * (yhln[t] + yhtn[t] + yhptn[t])
        "Expected growth rate of real dividends, for WPSN eq. (VAR exp.)"
        zdivgr[t] - zdivgr_a[t] = y_zdivgr[1] + y_zdivgr[2] * picnia[t] + y_zdivgr[3] * picnia[t - 1] + y_zdivgr[4] * picnia[t - 2] + y_zdivgr[5] * picnia[t - 3] + y_zdivgr[6] * rff[t] + y_zdivgr[7] * rff[t - 1] + y_zdivgr[8] * rff[t - 2] + y_zdivgr[9] * rff[t - 3] + y_zdivgr[10] * rtr[t] + y_zdivgr[11] * ptr[t] + y_zdivgr[12] * xgap[t] + y_zdivgr[13] * xgap[t - 1] + y_zdivgr[14] * xgap[t - 2] + y_zdivgr[15] * xgap[t - 3] + y_zdivgr[16] * hgynid[t] + y_zdivgr[17] * hgynid[t - 1] + y_zdivgr[18] * hgynid[t - 2] + y_zdivgr[19] * hgynid[t - 3] + y_zdivgr[20] * hxbt[t]
        "Expected growth rate of business output EBFI (VAR exp.)"
        zebfi[t] - zebfi_a[t] = y_zebfi[1] + y_zebfi[2] * picnia[t - 1] + y_zebfi[3] * picnia[t - 2] + y_zebfi[4] * picnia[t - 3] + y_zebfi[5] * picnia[t - 4] + y_zebfi[6] * rff[t - 1] + y_zebfi[7] * rff[t - 2] + y_zebfi[8] * rff[t - 3] + y_zebfi[9] * rff[t - 4] + y_zebfi[10] * rtr[t - 1] + y_zebfi[11] * ptr[t - 1] + y_zebfi[12] * xgap[t - 1] + y_zebfi[13] * xgap[t - 2] + y_zebfi[14] * xgap[t - 3] + y_zebfi[15] * xgap[t - 4] + y_zebfi[16] * @dlog(qebfi[t - 1]) + y_zebfi[17] * @dlog(qebfi[t - 2]) + y_zebfi[18] * @dlog(qebfi[t - 3]) + y_zebfi[19] * @dlog(qebfi[t - 4]) + (y_zebfi[20] * hxbt[t - 1]) / 400 + (y_zebfi[21] * hgpbfir[t - 1]) / 400
        "Expected growth rate of target durable consumption, for ECD eq. (VAR exp.)"
        zecd[t] - zecd_a[t] = y_zecd[1] * picnia[t - 1] + y_zecd[2] * picnia[t - 2] + y_zecd[3] * picnia[t - 3] + y_zecd[4] * picnia[t - 4] + y_zecd[5] * rff[t - 1] + y_zecd[6] * rff[t - 2] + y_zecd[7] * rff[t - 3] + y_zecd[8] * rff[t - 4] + y_zecd[9] * xgap2[t - 1] + y_zecd[10] * xgap2[t - 2] + y_zecd[11] * xgap2[t - 3] + y_zecd[12] * xgap2[t - 4] + y_zecd[13] * ptr[t - 1] + y_zecd[14] * rtr[t - 1] + y_zecd[15] * yhgap[t - 1] + y_zecd[16] * yhgap[t - 2] + y_zecd[17] * yhgap[t - 3] + y_zecd[18] * yhgap[t - 4] + y_zecd[19] * yhtgap[t - 1] + y_zecd[20] * yhtgap[t - 2] + y_zecd[21] * yhtgap[t - 3] + y_zecd[22] * yhtgap[t - 4] + y_zecd[23] * yhpgap[t - 1] + y_zecd[24] * yhpgap[t - 2] + y_zecd[25] * yhpgap[t - 3] + y_zecd[26] * yhpgap[t - 4] + (y_zecd[27] * hggdpt[t - 1]) / 400 + (y_zecd[28] * hgpcdr[t - 1]) / 400 + y_zecd[29] * @dlog(qecd[t - 1]) + y_zecd[30] * @dlog(qecd[t - 2]) + y_zecd[31] * @dlog(qecd[t - 3]) + y_zecd[32] * @dlog(qecd[t - 4])
        "Expected growth rate of target nondurables and nonhousing services, for ECO eq (VAR"
        zeco[t] - zeco_a[t] = y_zeco[1] * picnia[t - 1] + y_zeco[2] * picnia[t - 2] + y_zeco[3] * picnia[t - 3] + y_zeco[4] * picnia[t - 4] + y_zeco[5] * rff[t - 1] + y_zeco[6] * rff[t - 2] + y_zeco[7] * rff[t - 3] + y_zeco[8] * rff[t - 4] + y_zeco[9] * xgap2[t - 1] + y_zeco[10] * xgap2[t - 2] + y_zeco[11] * xgap2[t - 3] + y_zeco[12] * xgap2[t - 4] + y_zeco[13] * ptr[t - 1] + y_zeco[14] * rtr[t - 1] + y_zeco[15] * yhgap[t - 1] + y_zeco[16] * yhgap[t - 2] + y_zeco[17] * yhgap[t - 3] + y_zeco[18] * yhgap[t - 4] + y_zeco[19] * yhtgap[t - 1] + y_zeco[20] * yhtgap[t - 2] + y_zeco[21] * yhtgap[t - 3] + y_zeco[22] * yhtgap[t - 4] + y_zeco[23] * yhpgap[t - 1] + y_zeco[24] * yhpgap[t - 2] + y_zeco[25] * yhpgap[t - 3] + y_zeco[26] * yhpgap[t - 4] + (y_zeco[27] * hggdpt[t - 1]) / 400 + y_zeco[28] * @dlog(qeco[t - 1]) + y_zeco[29] * @dlog(qeco[t - 2]) + y_zeco[30] * @dlog(qeco[t - 3]) + y_zeco[31] * @dlog(qeco[t - 4])
        "Expected growth rate of target residential investment, for EH eq. (VAR exp.)"
        zeh[t] - zeh_a[t] = y_zeh[1] * picnia[t - 1] + y_zeh[2] * picnia[t - 2] + y_zeh[3] * picnia[t - 3] + y_zeh[4] * picnia[t - 4] + y_zeh[5] * rff[t - 1] + y_zeh[6] * rff[t - 2] + y_zeh[7] * rff[t - 3] + y_zeh[8] * rff[t - 4] + y_zeh[9] * xgap2[t - 1] + y_zeh[10] * xgap2[t - 2] + y_zeh[11] * xgap2[t - 3] + y_zeh[12] * xgap2[t - 4] + y_zeh[13] * ptr[t - 1] + y_zeh[14] * rtr[t - 1] + y_zeh[15] * yhgap[t - 1] + y_zeh[16] * yhgap[t - 2] + y_zeh[17] * yhgap[t - 3] + y_zeh[18] * yhgap[t - 4] + y_zeh[19] * yhtgap[t - 1] + y_zeh[20] * yhtgap[t - 2] + y_zeh[21] * yhtgap[t - 3] + y_zeh[22] * yhtgap[t - 4] + y_zeh[23] * yhpgap[t - 1] + y_zeh[24] * yhpgap[t - 2] + y_zeh[25] * yhpgap[t - 3] + y_zeh[26] * yhpgap[t - 4] + (y_zeh[27] * hggdpt[t - 1]) / 400 + y_zeh[28] * @dlog(qeh[t - 1]) + y_zeh[29] * @dlog(qeh[t - 2]) + y_zeh[30] * @dlog(qeh[t - 3]) + y_zeh[31] * @dlog(qeh[t - 4])
        "Expected output gap, for RG5 eq. (VAR exp.)"
        zgap05[t] - zgap05_a[t] = y_zgap05[1] + y_zgap05[2] * picnia[t] + y_zgap05[3] * picnia[t - 1] + y_zgap05[4] * picnia[t - 2] + y_zgap05[5] * picnia[t - 3] + y_zgap05[6] * rff[t] + y_zgap05[7] * rff[t - 1] + y_zgap05[8] * rff[t - 2] + y_zgap05[9] * rff[t - 3] + y_zgap05[10] * rtr[t] + y_zgap05[11] * ptr[t] + y_zgap05[12] * xgap[t] + y_zgap05[13] * xgap[t - 1] + y_zgap05[14] * xgap[t - 2] + y_zgap05[15] * xgap[t - 3]
        "Expected output gap, for RG10 eq. (VAR exp.)"
        zgap10[t] - zgap10_a[t] = y_zgap10[1] + y_zgap10[2] * picnia[t] + y_zgap10[3] * picnia[t - 1] + y_zgap10[4] * picnia[t - 2] + y_zgap10[5] * picnia[t - 3] + y_zgap10[6] * rff[t] + y_zgap10[7] * rff[t - 1] + y_zgap10[8] * rff[t - 2] + y_zgap10[9] * rff[t - 3] + y_zgap10[10] * rtr[t] + y_zgap10[11] * ptr[t] + y_zgap10[12] * xgap[t] + y_zgap10[13] * xgap[t - 1] + y_zgap10[14] * xgap[t - 2] + y_zgap10[15] * xgap[t - 3]
        "Expected output gap, for RG30 eq. (VAR exp.)"
        zgap30[t] - zgap30_a[t] = y_zgap30[1] + y_zgap30[2] * picnia[t] + y_zgap30[3] * picnia[t - 1] + y_zgap30[4] * picnia[t - 2] + y_zgap30[5] * picnia[t - 3] + y_zgap30[6] * rff[t] + y_zgap30[7] * rff[t - 1] + y_zgap30[8] * rff[t - 2] + y_zgap30[9] * rff[t - 3] + y_zgap30[10] * rtr[t] + y_zgap30[11] * ptr[t] + y_zgap30[12] * xgap[t] + y_zgap30[13] * xgap[t - 1] + y_zgap30[14] * xgap[t - 2] + y_zgap30[15] * xgap[t - 3]
        "Expected output gap, for ECD eq. (VAR exp.)"
        zgapc2[t] - zgapc2_a[t] = y_zgapc2[1] * picnia[t - 1] + y_zgapc2[2] * picnia[t - 2] + y_zgapc2[3] * picnia[t - 3] + y_zgapc2[4] * picnia[t - 4] + y_zgapc2[5] * rff[t - 1] + y_zgapc2[6] * rff[t - 2] + y_zgapc2[7] * rff[t - 3] + y_zgapc2[8] * rff[t - 4] + y_zgapc2[9] * xgap2[t - 1] + y_zgapc2[10] * xgap2[t - 2] + y_zgapc2[11] * xgap2[t - 3] + y_zgapc2[12] * xgap2[t - 4] + y_zgapc2[13] * ptr[t - 1] + y_zgapc2[14] * rtr[t - 1]
        "Expected growth rate of target aggregate hours (VAR exp.)"
        zlhp[t] - zlhp_a[t] = y_zlhp[1] * picnia[t - 1] + y_zlhp[2] * picnia[t - 2] + y_zlhp[3] * picnia[t - 3] + y_zlhp[4] * picnia[t - 4] + y_zlhp[5] * rff[t - 1] + y_zlhp[6] * rff[t - 2] + y_zlhp[7] * rff[t - 3] + y_zlhp[8] * rff[t - 4] + y_zlhp[9] * rtr[t - 1] + y_zlhp[10] * ptr[t - 1] + y_zlhp[11] * xgap[t - 1] + y_zlhp[12] * xgap[t - 2] + y_zlhp[13] * xgap[t - 3] + y_zlhp[14] * xgap[t - 4] + y_zlhp[15] * (@dlog(xbo[t - 1]) - @dlog(lprdt[t - 1])) + (y_zlhp[16] * (hlept[t - 1] + hqlww[t - 1])) / 400
        "Expected cons. price infl., for RCCH, RRMET, and YHPNTN eqs. (10-yr mat.) (VAR exp.)"
        zpi10[t] - zpi10_a[t] = y_zpi10[1] * picnia[t - 1] + y_zpi10[2] * picnia[t - 2] + y_zpi10[3] * picnia[t - 3] + y_zpi10[4] * picnia[t - 4] + y_zpi10[5] * rff[t - 1] + y_zpi10[6] * rff[t - 2] + y_zpi10[7] * rff[t - 3] + y_zpi10[8] * rff[t - 4] + y_zpi10[9] * rtr[t - 1] + y_zpi10[10] * ptr[t - 1] + y_zpi10[11] * xgap[t - 1] + y_zpi10[12] * xgap[t - 2] + y_zpi10[13] * xgap[t - 3] + y_zpi10[14] * xgap[t - 4]
        "Expected cons. price infl., for FPXR eq. (10-yr mat.) (VAR exp.)"
        zpi10f[t] - zpi10f_a[t] = zpi10[t]
        "Expected cons. price infl., for RCCD eq. (5-yr mat.) (VAR exp.)"
        zpi5[t] - zpi5_a[t] = y_zpi5[1] * picnia[t - 1] + y_zpi5[2] * picnia[t - 2] + y_zpi5[3] * picnia[t - 3] + y_zpi5[4] * picnia[t - 4] + y_zpi5[5] * rff[t - 1] + y_zpi5[6] * rff[t - 2] + y_zpi5[7] * rff[t - 3] + y_zpi5[8] * rff[t - 4] + y_zpi5[9] * rtr[t - 1] + y_zpi5[10] * ptr[t - 1] + y_zpi5[11] * xgap[t - 1] + y_zpi5[12] * xgap[t - 2] + y_zpi5[13] * xgap[t - 3] + y_zpi5[14] * xgap[t - 4]
        "Expected output price infl., for RPD eq. (5-yr mat.) (VAR exp.)"
        zpib5[t] - zpib5_a[t] = y_zpib5[1] + y_zpib5[2] * picnia[t - 1] + y_zpib5[3] * picnia[t - 2] + y_zpib5[4] * picnia[t - 3] + y_zpib5[5] * picnia[t - 4] + y_zpib5[6] * rff[t - 1] + y_zpib5[7] * rff[t - 2] + y_zpib5[8] * rff[t - 3] + y_zpib5[9] * rff[t - 4] + y_zpib5[10] * rtr[t - 1] + y_zpib5[11] * ptr[t - 1] + y_zpib5[12] * xgap[t - 1] + y_zpib5[13] * xgap[t - 2] + y_zpib5[14] * xgap[t - 3] + y_zpib5[15] * xgap[t - 4] + y_zpib5[16] * 400 * @dlog(pxb[t - 1]) + y_zpib5[17] * 400 * @dlog(pxb[t - 2]) + y_zpib5[18] * 400 * @dlog(pxb[t - 3]) + y_zpib5[19] * 400 * @dlog(pxb[t - 4])
        "Expected cons. price infl., for REQ eq. (30-yr mat.) (VAR exp.)"
        zpic30[t] - zpic30_a[t] = y_zpic30[1] + y_zpic30[2] * picnia[t] + y_zpic30[3] * picnia[t - 1] + y_zpic30[4] * picnia[t - 2] + y_zpic30[5] * picnia[t - 3] + y_zpic30[6] * rff[t] + y_zpic30[7] * rff[t - 1] + y_zpic30[8] * rff[t - 2] + y_zpic30[9] * rff[t - 3] + y_zpic30[10] * rtr[t] + y_zpic30[11] * ptr[t] + y_zpic30[12] * xgap[t] + y_zpic30[13] * xgap[t - 1] + y_zpic30[14] * xgap[t - 2] + y_zpic30[15] * xgap[t - 3]
        "Expected 4-qtr consumer price inflation (8 qtrs. in the future) (VAR exp.)"
        zpic58[t] - zpic58_a[t] = y_zpic58[1] * picnia[t] + y_zpic58[2] * picnia[t - 1] + y_zpic58[3] * picnia[t - 2] + y_zpic58[4] * picnia[t - 3] + y_zpic58[5] * rff[t] + y_zpic58[6] * rff[t - 1] + y_zpic58[7] * rff[t - 2] + y_zpic58[8] * rff[t - 3] + y_zpic58[9] * rtr[t] + y_zpic58[10] * ptr[t] + y_zpic58[11] * xgap[t] + y_zpic58[12] * xgap[t - 1] + y_zpic58[13] * xgap[t - 2] + y_zpic58[14] * xgap[t - 3]
        "Expected value of picxfe in the next quarter (VAR exp.)"
        zpicxfe[t] - zpicxfe_a[t] = (y_zpicxfe[1] * picxfe[t - 1] + y_zpicxfe[2] * picxfe[t - 2] + y_zpicxfe[3] * picxfe[t - 3] + y_zpicxfe[4] * picxfe[t - 4]) + (y_zpicxfe[5] * pieci[t - 1] + y_zpicxfe[6] * pieci[t - 2] + y_zpicxfe[7] * pieci[t - 3] + y_zpicxfe[8] * pieci[t - 4]) + (y_zpicxfe[9] * rff[t - 1] + y_zpicxfe[10] * rff[t - 2] + y_zpicxfe[11] * rff[t - 3] + y_zpicxfe[12] * rff[t - 4]) + (y_zpicxfe[13] * xgap2[t - 1] + y_zpicxfe[14] * xgap2[t - 2] + y_zpicxfe[15] * xgap2[t - 3] + y_zpicxfe[16] * xgap2[t - 4]) + y_zpicxfe[17] * rtr[t - 1] + y_zpicxfe[18] * ptr[t - 1] + y_zpicxfe[19] * log(qpcnia[t - 1] / pcnia[t - 1]) + y_zpicxfe[20] * log(qpl[t - 1] / pl[t - 1]) + y_zpicxfe[21] * (hlprdt[t - 1] - 400 * huqpct[t - 1]) + (y_zpicxfe[22] * (lur[t - 1] - lurnat[t - 1]) + y_zpicxfe[23] * (lur[t - 2] - lurnat[t - 2]))
        "Expected value of pieci in the next quarter (VAR exp.)"
        zpieci[t] - zpieci_a[t] = (y_zpieci[1] * picxfe[t - 1] + y_zpieci[2] * picxfe[t - 2] + y_zpieci[3] * picxfe[t - 3] + y_zpieci[4] * picxfe[t - 4]) + (y_zpieci[5] * pieci[t - 1] + y_zpieci[6] * pieci[t - 2] + y_zpieci[7] * pieci[t - 3] + y_zpieci[8] * pieci[t - 4]) + (y_zpieci[9] * rff[t - 1] + y_zpieci[10] * rff[t - 2] + y_zpieci[11] * rff[t - 3] + y_zpieci[12] * rff[t - 4]) + (y_zpieci[13] * xgap2[t - 1] + y_zpieci[14] * xgap2[t - 2] + y_zpieci[15] * xgap2[t - 3] + y_zpieci[16] * xgap2[t - 4]) + y_zpieci[17] * rtr[t - 1] + y_zpieci[18] * ptr[t - 1] + y_zpieci[19] * log(qpcnia[t - 1] / pcnia[t - 1]) + y_zpieci[20] * log(qpl[t - 1] / pl[t - 1]) + y_zpieci[21] * (hlprdt[t - 1] - 400 * huqpct[t - 1]) + (y_zpieci[22] * (lur[t - 1] - lurnat[t - 1]) + y_zpieci[23] * (lur[t - 2] - lurnat[t - 2]))
        "Expected federal funds rate, for RG10 eq. (10-yr mat.) (VAR exp.)"
        zrff10[t] - zrff10_a[t] = y_zrff10[1] + y_zrff10[2] * picnia[t] + y_zrff10[3] * picnia[t - 1] + y_zrff10[4] * picnia[t - 2] + y_zrff10[5] * picnia[t - 3] + y_zrff10[6] * rff[t] + y_zrff10[7] * rff[t - 1] + y_zrff10[8] * rff[t - 2] + y_zrff10[9] * rff[t - 3] + y_zrff10[10] * rtr[t] + y_zrff10[11] * ptr[t] + y_zrff10[12] * xgap[t] + y_zrff10[13] * xgap[t - 1] + y_zrff10[14] * xgap[t - 2] + y_zrff10[15] * xgap[t - 3]
        "Expected federal funds rate, for RG30 eq. (30-yr mat.) (VAR exp.)"
        zrff30[t] - zrff30_a[t] = y_zrff30[1] + y_zrff30[2] * picnia[t] + y_zrff30[3] * picnia[t - 1] + y_zrff30[4] * picnia[t - 2] + y_zrff30[5] * picnia[t - 3] + y_zrff30[6] * rff[t] + y_zrff30[7] * rff[t - 1] + y_zrff30[8] * rff[t - 2] + y_zrff30[9] * rff[t - 3] + y_zrff30[10] * rtr[t] + y_zrff30[11] * ptr[t] + y_zrff30[12] * xgap[t] + y_zrff30[13] * xgap[t - 1] + y_zrff30[14] * xgap[t - 2] + y_zrff30[15] * xgap[t - 3]
        "Expected federal funds rate, for RG5 eq. (5-yr mat.) (VAR exp.)"
        zrff5[t] - zrff5_a[t] = y_zrff5[1] + y_zrff5[2] * picnia[t] + y_zrff5[3] * picnia[t - 1] + y_zrff5[4] * picnia[t - 2] + y_zrff5[5] * picnia[t - 3] + y_zrff5[6] * rff[t] + y_zrff5[7] * rff[t - 1] + y_zrff5[8] * rff[t - 2] + y_zrff5[9] * rff[t - 3] + y_zrff5[10] * rtr[t] + y_zrff5[11] * ptr[t] + y_zrff5[12] * xgap[t] + y_zrff5[13] * xgap[t - 1] + y_zrff5[14] * xgap[t - 2] + y_zrff5[15] * xgap[t - 3]
        "Expected level of real after-tax household income, for QEC eq. (VAR exp.)"
        log(zyh[t]) - zyh_a[t] = y_zyh[1] * picnia[t] + y_zyh[2] * picnia[t - 1] + y_zyh[3] * picnia[t - 2] + y_zyh[4] * picnia[t - 3] + y_zyh[5] * rff[t] + y_zyh[6] * rff[t - 1] + y_zyh[7] * rff[t - 2] + y_zyh[8] * rff[t - 3] + y_zyh[9] * xgap2[t] + y_zyh[10] * xgap2[t - 1] + y_zyh[11] * xgap2[t - 2] + y_zyh[12] * xgap2[t - 3] + y_zyh[13] * ptr[t] + y_zyh[14] * rtr[t] + y_zyh[15] * yhgap[t] + y_zyh[16] * yhgap[t - 1] + y_zyh[17] * yhgap[t - 2] + y_zyh[18] * yhgap[t - 3] + y_zyh[19] * log(zyhst[t] * xgdpt[t])
        "Expected level of real after-tax property income, for QEC eq. (VAR exp.)"
        log(zyhp[t]) - zyhp_a[t] = y_zyhp[1] * picnia[t] + y_zyhp[2] * picnia[t - 1] + y_zyhp[3] * picnia[t - 2] + y_zyhp[4] * picnia[t - 3] + y_zyhp[5] * rff[t] + y_zyhp[6] * rff[t - 1] + y_zyhp[7] * rff[t - 2] + y_zyhp[8] * rff[t - 3] + y_zyhp[9] * xgap2[t] + y_zyhp[10] * xgap2[t - 1] + y_zyhp[11] * xgap2[t - 2] + y_zyhp[12] * xgap2[t - 3] + y_zyhp[13] * ptr[t] + y_zyhp[14] * rtr[t] + y_zyhp[15] * yhgap[t] + y_zyhp[16] * yhgap[t - 1] + y_zyhp[17] * yhgap[t - 2] + y_zyhp[18] * yhgap[t - 3] + y_zyhp[19] * yhpgap[t] + y_zyhp[20] * yhpgap[t - 1] + y_zyhp[21] * yhpgap[t - 2] + y_zyhp[22] * yhpgap[t - 3] + y_zyhp[23] * log(zyhpst[t] * zyhst[t] * xgdpt[t])
        "Expected level of real transfer income, for QEC eq. (VAR exp.)"
        log(zyht[t]) - zyht_a[t] = y_zyht[1] * picnia[t] + y_zyht[2] * picnia[t - 1] + y_zyht[3] * picnia[t - 2] + y_zyht[4] * picnia[t - 3] + y_zyht[5] * rff[t] + y_zyht[6] * rff[t - 1] + y_zyht[7] * rff[t - 2] + y_zyht[8] * rff[t - 3] + y_zyht[9] * xgap2[t] + y_zyht[10] * xgap2[t - 1] + y_zyht[11] * xgap2[t - 2] + y_zyht[12] * xgap2[t - 3] + y_zyht[13] * ptr[t] + y_zyht[14] * rtr[t] + y_zyht[15] * yhgap[t] + y_zyht[16] * yhgap[t - 1] + y_zyht[17] * yhgap[t - 2] + y_zyht[18] * yhgap[t - 3] + y_zyht[19] * yhtgap[t] + y_zyht[20] * yhtgap[t - 1] + y_zyht[21] * yhtgap[t - 2] + y_zyht[22] * yhtgap[t - 3] + y_zyht[23] * log(zyhtst[t] * zyhst[t] * xgdpt[t])
        "Expected rate of growth of target real dividends, for YNIDN eq. (VAR exp.)"
        zynid[t] - zynid_a[t] = y_zynid[1] + y_zynid[2] * picnia[t - 1] + y_zynid[3] * picnia[t - 2] + y_zynid[4] * picnia[t - 3] + y_zynid[5] * picnia[t - 4] + y_zynid[6] * rff[t - 1] + y_zynid[7] * rff[t - 2] + y_zynid[8] * rff[t - 3] + y_zynid[9] * rff[t - 4] + y_zynid[10] * rtr[t - 1] + y_zynid[11] * ptr[t - 1] + y_zynid[12] * xgap[t - 1] + y_zynid[13] * xgap[t - 2] + y_zynid[14] * xgap[t - 3] + y_zynid[15] * xgap[t - 4] + y_zynid[16] * @dlog(qynidn[t - 1] / pxb[t - 1]) + y_zynid[17] * @dlog(qynidn[t - 2] / pxb[t - 2]) + y_zynid[18] * @dlog(qynidn[t - 3] / pxb[t - 3]) + y_zynid[19] * @dlog(qynidn[t - 4] / pxb[t - 4]) + (y_zynid[20] * hggdpt[t - 1]) / 400

        # Behavioral equations:
        "Monetary policy indicator for unemployment threshold"
        dmptlur[t] - dmptlur_a[t] = heaviside(-(y_dmptlur[1] * (lur[t] - lurtrsh[t])))
        "Monetary policy indicator for both thresholds"
        dmptmax[t] - dmptmax_a[t] = max(dmptlur[t], dmptpi[t])
        "Monetary policy indicator for inflation threshold"
        dmptpi[t] - dmptpi_a[t] = heaviside(-(y_dmptpi[1] * (zpic58[t] - pitrsh[t])))
        "Monetary policy indicator for policy rule thresholds"
        dmptr[t] - dmptr_a[t] = max(dmptmax[t], dmptr[t - 1])
        "Price inflation aggregation adjustment"
        (dpadj[t] - dpadj_a[t]) - dpadj[t - 1] = y_dpadj[1] * dpgap[t - 1]
        "Business Fixed Investment, cw 2012\$"
        @dlog(ebfi[t]) - ebfi_a[t] = (y_ebfi[1] * log(qebfi[t - 1] / ebfi[t - 1]) + y_ebfi[2] * @dlog(ebfi[t - 1]) + y_ebfi[3] * @dlog(ebfi[t - 2]) + y_ebfi[4] * zebfi[t]) * (1 - y_ebfi[5]) + y_ebfi[5] * (@dlog(xb[t - 1]) - hgpbfir[t - 1] / 400)
        "Consumption, cw 2012\$ (FRB/US definition)"
        log(ec[t]) - ec_a[t] = log(ec[t - 1]) + y_ec[1] * log(eco[t] / eco[t - 1]) + y_ec[2] * log(ech[t] / ech[t - 1]) + y_ec[3] * log((yhpcd[t] + jkcd[t]) / (yhpcd[t - 1] + jkcd[t - 1]))
        "Consumer expenditures on durable goods, cw 2012\$"
        @dlog(ecd[t]) - ecd_a[t] = y_ecd[1] * log(qecd[t - 1] / ecd[t - 1]) + y_ecd[2] * @dlog(ecd[t - 1]) + y_ecd[3] * zecd[t] + (y_ecd[4] * zgapc2[t]) / 400
        "Consumer expenditures on housing services, cw 2012\$"
        @d(ech[t] / kh[t - 1]) - ech_a[t] = y_ech[1] + (y_ech[2] * ech[t - 1]) / kh[t - 2] + y_ech[3] * @d(ech[t - 1] / kh[t - 2])
        "Personal consumption expenditures, cw 2012\$ (NIPA definition)"
        log(ecnia[t]) - ecnia_a[t] = log(ecnia[t - 1]) + y_ecnia[1] * log(eco[t] / eco[t - 1]) + y_ecnia[2] * log(ecd[t] / ecd[t - 1]) + y_ecnia[3] * log(ech[t] / ech[t - 1])
        "Consumer expenditures on non-durable goods and non-housing services, cw 2012\$"
        @dlog(eco[t]) - eco_a[t] = (y_eco[1] * log(qeco[t - 1] / eco[t - 1]) + y_eco[2] * @dlog(eco[t - 1]) + y_eco[3] * zeco[t]) * (1 - y_eco[4]) + y_eco[4] * @dlog(yhl[t] + yht[t])
        "Federal Government expenditures, CW 2012\$"
        @d(log(egfe[t]), 0, 1) - egfe_a[t] = y_egfe[1] + y_egfe[2] * log(egfe[t - 1] / egfet[t - 1]) + y_egfe[3] * @d(log(egfe[t - 1]), 0, 1) + y_egfe[4] * @d(log(egfe[t - 2]), 0, 1) + y_egfe[5] * @d(log(egfet[t]), 0, 1) + y_egfe[6] * xgap2[t] + y_egfe[7] * xgap2[t - 1]
        "Federal Government expenditures, CW 2012\$, Trend"
        @d(log(egfet[t]), 0, 1) - egfet_a[t] = y_egfet[1] + y_egfet[2] * log((0.01 * pegfr[t - 1] * pxp[t - 1] * egfet[t - 1]) / xgdptn[t - 1]) + (y_egfet[3] * (hggdpt[t] + hggdpt[t - 1] + hggdpt[t - 2] + hggdpt[t - 3])) / 1600
        "Federal government employee compensation, cw 2012\$"
        @d(log(egfl[t]), 0, 1) - egfl_a[t] = y_egfl[1] + y_egfl[2] * log(egfl[t - 1] / egflt[t - 1]) + y_egfl[3] * @d(log(egfl[t - 1]), 0, 1) + y_egfl[4] * @d(log(egfl[t - 2]), 0, 1) + y_egfl[5] * @d(log(egflt[t]), 0, 1) + y_egfl[6] * xgap2[t] + y_egfl[7] * xgap2[t - 1]
        "Federal government employee compensation, cw 2012\$, trend"
        @d(log(egflt[t]), 0, 1) - egflt_a[t] = y_egflt[1] + y_egflt[2] * log((0.01 * pgfl[t - 1] * egflt[t - 1]) / xgdptn[t - 1]) + (y_egflt[3] * (hggdpt[t] + hggdpt[t - 1] + hggdpt[t - 2] + hggdpt[t - 3])) / 1600
        "S&L Government expenditures, CW 2012\$"
        @d(log(egse[t]), 0, 1) - egse_a[t] = y_egse[1] + y_egse[2] * log(egse[t - 1] / egset[t - 1]) + y_egse[3] * @d(log(egse[t - 1]), 0, 1) + y_egse[4] * @d(log(egse[t - 2]), 0, 1) + y_egse[5] * @d(log(egset[t]), 0, 1) + y_egse[6] * xgap2[t] + y_egse[7] * xgap2[t - 1]
        "S&L Government expenditures, CW 2012\$, Trend"
        @d(log(egset[t]), 0, 1) - egset_a[t] = y_egset[1] + y_egset[2] * log((0.01 * pegsr[t - 1] * pxp[t - 1] * egset[t - 1]) / xgdptn[t - 1]) + (y_egset[3] * (hggdpt[t] + hggdpt[t - 1] + hggdpt[t - 2] + hggdpt[t - 3])) / 1600
        "S&L government employee compensation, cw 2012\$"
        @d(log(egsl[t]), 0, 1) - egsl_a[t] = y_egsl[1] + y_egsl[2] * log(egsl[t - 1] / egslt[t - 1]) + y_egsl[3] * @d(log(egsl[t - 1]), 0, 1) + y_egsl[4] * @d(log(egsl[t - 2]), 0, 1) + y_egsl[5] * @d(log(egslt[t]), 0, 1) + y_egsl[6] * xgap2[t] + y_egsl[7] * xgap2[t - 1]
        "S&L government employee compensation, cw 2012\$, trend"
        @d(log(egslt[t]), 0, 1) - egslt_a[t] = y_egslt[1] + y_egslt[2] * log((0.01 * pgsl[t - 1] * egslt[t - 1]) / xgdptn[t - 1]) + (y_egslt[3] * (hggdpt[t] + hggdpt[t - 1] + hggdpt[t - 2] + hggdpt[t - 3])) / 1600
        "Residential investment expenditures, cw 2012\$"
        @dlog(eh[t]) - eh_a[t] = y_eh[1] * log(qeh[t - 1] / eh[t - 1]) + y_eh[2] * @dlog(eh[t - 1]) + y_eh[3] * @dlog(eh[t - 2]) + y_eh[4] * zeh[t] + y_eh[5] * @d(rme[t - 1]) + y_eh[6] * d83[t] * @d(rme[t - 1])
        "Change in business inventories, current \$"
        ein[t] - ein_a[t] = 0.01 * pxp[t] * pkir[t] * ei[t]
        "Imports of goods and services ex. petroleum, cw 2012\$"
        @dlog(emo[t]) - emo_a[t] = y_emo[1] + y_emo[2] * log((emo[t - 1] * (pmo[t - 1] / 100)) / (uemot[t - 1] * xgdpn[t - 1])) + (y_emo[3] * (xgap2[t] - xgap2[t - 1])) / 100 + (y_emo[4] * (xgap2[t - 1] - xgap2[t - 2])) / 100 + y_emo[5] * log(ddockm[t]) + y_emo[6] * @dlog(ddockm[t])
        "Petroleum imports, cw 2012\$"
        log(emp[t] / xgdp[t]) - emp_a[t] = log(emptrt[t]) + y_emp[1] * @d(log(pmp[t] / pxb[t])) + y_emp[2] * xgap2[t - 1]
        "Exports of goods and services, cw 2012\$"
        @dlog(ex[t]) - ex_a[t] = y_ex[1] + y_ex[2] * log((ex[t - 1] * (pxr[t - 1] * pxp[t - 1] * fpx[t - 1])) / (fgdp[t - 1] * fpc[t - 1])) + (y_ex[3] * (fxgap[t] - fxgap[t - 1])) / 100 + (y_ex[4] * (fxgap[t - 1] - fxgap[t - 2])) / 100 + y_ex[5] * ddockx[t]
        "Foreign aggregate GDP (world, bilateral export weights), trend"
        @d(log(fgdpt[t]), 0, 1) - fgdpt_a[t] = y_fgdpt[1] + y_fgdpt[2] * log(fgdpt[t - 1] / xgdpt[t - 1]) + (y_fgdpt[3] * (hggdpt[t] + hggdpt[t - 1] + hggdpt[t - 2] + hggdpt[t - 3])) / 1600
        "Foreign consumer price inflation (G10)"
        fpi10[t] - fpi10_a[t] = y_fpi10[1] * ((fpi10[t - 1] + fpi10[t - 2] + fpi10[t - 3] + fpi10[t - 4]) / 4) + y_fpi10[2] * fpitrg[t] + y_fpi10[3] * fxgap[t - 1]
        "Foreign consumer price inflation, trend (G10)"
        fpi10t[t] - fpi10t_a[t] = y_fpi10t[1] * fpi10t[t - 1] + y_fpi10t[2] * fpi10[t]
        "Foreign consumer price inflation (G39, bilateral export trade weights)"
        fpic[t] - fpic_a[t] = y_fpic[1] + y_fpic[2] * fpi10[t] + y_fpic[3] * fpic[t - 1]
        "Real exchange rate (G39, import/export trade weights)"
        (log(fpxr[t]) - fpxr_a[t]) - log(fpxrr[t]) = y_fpxr[1] * (((rg10[t] - zpi10f[t]) - frl10[t]) + fpi10t[t]) + y_fpxr[2] * (fnin[t] / xgdpn[t])
        "Real exchange rate residual"
        @d(log(fpxrr[t]), 0, 1) - fpxrr_a[t] = y_fpxrr[1] * log(fpxrrt[t - 1] / fpxrr[t - 1]) + y_fpxrr[2] * @d(log(fpxrr[t - 1]), 0, 1) + (1 - y_fpxrr[2]) * @d(log(fpxrrt[t]), 0, 1)
        "Foreign long-term interest rate (G10)"
        (frl10[t] - frl10[t - 1]) - frl10_a[t] = y_frl10[1] + y_frl10[2] * (frl10[t - 1] - frs10[t - 1]) + y_frl10[3] * (frl10[t - 1] - frl10[t - 2]) + y_frl10[4] * (frs10[t] - frs10[t - 1]) + y_frl10[5] * (fxgap[t] - fxgap[t - 1])
        "Foreign short-term interest rate (G10)"
        frs10[t] - frs10_a[t] = dfmprr[t] * (y_frs10[1] + y_frs10[2] * frstar[t - 1] + y_frs10[3] * ((fpi10[t] + fpi10[t - 1] + fpi10[t - 2] + fpi10[t - 3]) / 4) + y_frs10[4] * ((fpi10[t] + fpi10[t - 1] + fpi10[t - 2] + fpi10[t - 3]) / 4 - fpitrg[t]) + y_frs10[5] * fxgap[t]) + (1 - dfmprr[t]) * (rfrs10[t] + (fpi10[t] + fpi10[t - 1] + fpi10[t - 2] + fpi10[t - 3]) / 4)
        "Equilibrium real short-term interest rate used in foreign Taylor rule"
        frstar[t] - frstar_a[t] = y_frstar[1] * frstar[t - 1] + y_frstar[2] * (frs10[t] - (fpi10[t] + fpi10[t - 1] + fpi10[t - 2] + fpi10[t - 3]) / 4)
        "Foreign output gap (world, bilateral export weights)"
        fxgap[t] - fxgap_a[t] = +(y_fxgap[1]) * fxgap[t - 1] + y_fxgap[2] * fxgap[t - 2] + y_fxgap[3] * ((((((frs10[t - 1] - (fpi10[t - 1] + fpi10[t - 2] + fpi10[t - 3] + fpi10[t - 4]) / 4) + frs10[t - 2]) - (fpi10[t - 2] + fpi10[t - 3] + fpi10[t - 4] + fpi10[t - 5]) / 4) + frs10[t - 3]) - (fpi10[t - 3] + fpi10[t - 4] + fpi10[t - 5] + fpi10[t - 6]) / 4) / 3 - frstar[t]) + y_fxgap[4] * xgap2[t - 1]
        "Federal government net transfer payments, current \$"
        gtn[t] - gtn_a[t] = 0.01 * pgdp[t] * gtr[t]
        "Government net transfer payments, deflated by PGDP"
        gtr[t] - gtr_a[t] = (gtrd[t] + gtrt[t]) * xgdpt[t]
        "Deviation of ratio of government transfers to GDP from trend ratio"
        gtrd[t] - gtrd_a[t] = y_gtrd[2] * xgap2[t] + (y_gtrd[3] * (xgap2[t - 1] + xgap2[t - 2] + xgap2[t - 3] + xgap2[t - 4])) / 4 + y_gtrd[1] * ((gtrd[t - 1] - y_gtrd[2] * xgap2[t - 1]) - (y_gtrd[3] * (xgap2[t - 2] + xgap2[t - 3] + xgap2[t - 4] + xgap2[t - 5])) / 4)
        "Petroleum imports, cw 2012\$, trend growth rate"
        hgemp[t] - hgemp_a[t] = y_hgemp[1] * hgemp[t - 1] + y_hgemp[2] * 400 * log(emp[t] / emp[t - 1])
        "Trend growth rate of PKIR"
        hgpkir[t] - hgpkir_a[t] = y_hgpkir[1] * hgpkir[t - 1] + y_hgpkir[2] * 400 * log(pkir[t] / pkir[t - 1])
        "Growth rate of KS, cw 2012\$ (compound annual rate)"
        hks[t] - hks_a[t] = (400 * (ykbfin[t] * @d(log(kbfi[t]), 0, 1) + ykin[t] * @d(log(ki[t]), 0, 1))) / (ykbfin[t] + ykin[t]) + hksr[t]
        "Trend growth rate of LEP (annual rate)"
        hlept[t] - hlept_a[t] = 400 * hqlfpr[t] + 400 * @d(log(n16[t]), 0, 1)
        "Trend growth rate of multifactor productivity"
        hmfpt[t] - hmfpt_a[t] = y_hmfpt[1] + y_hmfpt[2] * hmfpt[t - 1]
        "Drift component of change in QLFPR"
        hqlfpr[t] - hqlfpr_a[t] = y_hqlfpr[1] + y_hqlfpr[2] * hqlfpr[t - 1]
        "Trend growth rate of workweek"
        hqlww[t] - hqlww_a[t] = y_hqlww[1] * hqlww[t - 1] + (1 - y_hqlww[1]) * y_hqlww[2]
        "Drift term in stochastic component of trend ratio of PCNIA to PXP"
        huqpct[t] - huqpct_a[t] = y_huqpct[1] + y_huqpct[2] * huqpct[t - 1]
        "Drift term in UXBT"
        huxb[t] - huxb_a[t] = (1 - dglprd[t]) * (y_huxb[1] + y_huxb[2] * huxb[t - 1])
        "Residual Factor (Trend rate of growth of XB)"
        hxbtr[t] - hxbtr_a[t] = 0
        "Consumption of fixed capital, current \$"
        jccan[t] / xgdpn[t] - jccan_a[t] = y_jccan[1] * (1 - y_jccan[2]) + (y_jccan[2] * jccan[t - 1]) / xgdpn[t - 1] + ((1 - y_jccan[2]) * pkbfir[t - 1] * kbfi[t - 1] * jrbfi[t] * pxp[t - 1] * 0.01) / xgdpn[t - 1]
        "Stock of private inventories, cw 2012\$"
        @dlog(ki[t]) - ki_a[t] = y_ki[5] + y_ki[1] * (log(qkir[t]) - log(ki[t - 1] / xfs[t - 1])) + y_ki[2] * (@dlog(ki[t - 1]) - y_ki[5]) + y_ki[3] * @dlog(xfs[t - 1]) + y_ki[4] * @dlog(xfs[t - 2])
        "Difference between household and business sector payroll employment, less government employment"
        log(leo[t]) - leo_a[t] = log(qleor[t] * qlf[t]) + y_leo[1] * log(leo[t - 1] / (qleor[t - 1] * qlf[t - 1])) + y_leo[2] * xgap2[t - 1]
        "Labor force participation rate"
        (@d(lfpr[t]) - hqlfpr[t]) - lfpr_a[t] = y_lfpr[1] * (qlfpr[t - 1] - lfpr[t - 1]) + y_lfpr[2] * (lur[t - 1] - lurnat[t - 1])
        "Aggregate labor hours,  business sector (employee and  self-employed)"
        @dlog(lhp[t]) - lhp_a[t] = y_lhp[1] * log(qlhp[t - 1] / lhp[t - 1]) + y_lhp[2] * @dlog(lhp[t - 1]) + y_lhp[3] * zlhp[t] + y_lhp[4] * (@dlog(xbo[t]) - hlprdt[t - 1] / 400) + y_lhp[5] * (@dlog(xbo[t - 1]) - hlprdt[t - 2] / 400)
        "Civilian unemployment rate (break adjusted)"
        lur[t] - lur_a[t] = 100 * (1 - leh[t] / lf[t])
        "Natural rate of unemployment"
        lurnat[t] - lurnat_a[t] = lurnat[t - 1]
        "Workweek,  business sector (employee and self-employed)"
        (@dlog(lww[t]) - hqlww[t] / 400) - lww_a[t] = y_lww[1] * log(qlww[t - 1] / lww[t - 1]) + y_lww[2] * (@dlog(lhp[t]) - (hlept[t] + hqlww[t]) / 400)
        "Multiplicative discrepancy for the difference between XGDI and XGDO"
        log(mei[t]) - mei_a[t] = y_mei[1] * log(mei[t - 1])
        "Multiplicative discrepancy for the difference between XGDP and XGDO"
        log(mep[t]) - mep_a[t] = y_mep[1] * log(mep[t - 1])
        "Multifactor productivity, trend level"
        log(mfpt[t]) - mfpt_a[t] = y_mfpt[1] + log(mfpt[t - 1]) + hmfpt[t] / 400
        "Price level of BFI compared to PXP"
        log(pbfir[t]) - pbfir_a[t] = (log(pbfir[t - 1]) + pipxnc[t] / 400 + dpadj[t]) - @d(log(pxp[t]), 0, 1)
        "Price index for consumer durables, cw (relative to to PCNIA)"
        @d(log(pcdr[t])) - pcdr_a[t] = y_pcdr[1] + y_pcdr[2] * @d(log(pcdr[t - 1]))
        "Price index for personal consumption expenditures on energy (relative to PCXFE)"
        @d(log(pcer[t]), 0, 1) - pcer_a[t] = y_pcer[1] * @d(log(pmp[t] / pcxfe[t]))
        "Price index for personal consumption expenditures on food (relative to PCXFE)"
        @d(log(pcfr[t]), 0, 1) - pcfr_a[t] = y_pcfr[1] * log(pcfr[t - 1] / pcfrt[t - 1]) + y_pcfr[2] + y_pcfr[3] * @d(log(pcfr[t - 1]), 0, 1) + y_pcfr[4] * @d(log(pcfr[t - 2]), 0, 1) + y_pcfr[5] * @d(log(pcfr[t - 3]), 0, 1) + y_pcfr[6] * @d(log(pcfrt[t]), 0, 1)
        "Price index for housing services, cw (relative to to PCNIA)"
        @d(log(pchr[t])) - pchr_a[t] = y_pchr[1] + y_pchr[2] * @d(log(pchr[t - 1]))
        "Price index for non-durable goods and non-housing services, cw (relative to to PCNIA"
        log(pcor[t]) - pcor_a[t] = log(pcor[t - 1]) + y_pcor[1] * log(pcdr[t] / pcdr[t - 1]) + y_pcor[2] * log(pchr[t] / pchr[t - 1])
        "Price index for personal consumption expendits ex. food and energy, cw (NIPA definit"
        @d(log(pcxfe[t]), 0, 1) - pcxfe_a[t] = picxfe[t] / 400
        "Price index for federal government expenditures, CW (relative to PXP)"
        (log(pegfr[t]) - pegfr_a[t]) - log(pegfr[t - 1]) = (pipxnc[t] / 400 + dpadj[t]) - @d(log(pxp[t]))
        "Price index for S&L government expenditures, CW (relative to PXP)"
        log(pegsr[t]) - pegsr_a[t] = (log(pegsr[t - 1]) + pipxnc[t] / 400 + dpadj[t]) - @d(log(pxp[t]))
        "Loan Performance House Price Index"
        @dlog(phouse[t]) - phouse_a[t] = y_phouse[1] + y_phouse[2] * @dlog(phouse[t - 1]) + y_phouse[3] * log(phouse[t - 1] / (pchr[t - 1] * pcnia[t - 1]))
        "Price index for residential investment, cw (relative to PXP)"
        (log(phr[t]) - phr_a[t]) - log(phr[t - 1]) = (y_phr[1] + pipxnc[t] / 400 + dpadj[t]) - @d(log(pxp[t]), 0, 1)
        "Inflation rate, personal consumption expenditures, cw"
        picnia[t] - picnia_a[t] = picxfe[t] + y_picnia[1] * 400 * (log(pcer[t]) - log(pcer[t - 1])) + y_picnia[2] * 400 * (log(pcfr[t]) - log(pcfr[t - 1]))
        "Inflation rate, personal consumption expenditures, ex. food and energy, cw"
        picxfe[t] - picxfe_a[t] = (y_picxfe[1] * picxfe[t - 1] + y_picxfe[3] * zpicxfe[t] + (1 - y_picxfe[3]) * (1 - y_picxfe[1]) * ptr[t - 1] + y_picxfe[2] * 400 * log(qpcnia[t - 1] / pcnia[t - 1])) / (1 + y_picxfe[1] * y_picxfe[3])
        "Annualized rate of growth of EI hourly compensation"
        pieci[t] - pieci_a[t] = (0.25 * y_pieci[1] * ((1 - y_pieci[4]) * (pieci[t - 1] + pieci[t - 2] + pieci[t - 3]) + pieci[t - 4]) + y_pieci[4] * zpieci[t] + (1 - y_pieci[4]) * (1 - y_pieci[1]) * ((ptr[t - 1] + hlprdt[t - 1]) - 400 * huqpct[t - 1]) + y_pieci[2] * (lur[t - 1] - lurnat[t - 1]) + y_pieci[3] * 400 * log(qpl[t - 1] / pl[t - 1])) / (1 + 0.25 * y_pieci[1] * y_pieci[4])
        "Rate of growth of PL"
        pipl[t] - pipl_a[t] = pieci[t]
        "Inflation rate, price of adjusted final sales excluding consumption (annual rate)"
        pipxnc[t] - pipxnc_a[t] = (picnia[t] - 1.99 * 400 * huqpct[t]) + y_pipxnc[1] * ((pipxnc[t - 1] - picnia[t - 1]) + 1.99 * 400 * huqpct[t - 1]) + y_pipxnc[2] * ((pipxnc[t - 2] - picnia[t - 2]) + 1.99 * 400 * huqpct[t - 2]) + y_pipxnc[3] * y_pipxnc[4] * 400 * (log(fpxr[t]) - log(fpxr[t - 1])) + 0.025 * 400 * log(qpxnc[t - 1] / pxnc[t - 1])
        "Price index for imports ex. petroleum, cw"
        @dlog(pmo[t]) - pmo_a[t] = y_pmo[1] + y_pmo[2] * ((log(qpmo[t]) + 0.64 * log(fpc[t - 1] / fpx[t - 1]) + 0.36 * log(pxb[t - 1])) - log(pmo[t - 1])) + y_pmo[3] * @dlog(fpc[t] / fpx[t]) + (1 - y_pmo[3]) * @dlog(pxb[t])
        "Price of imported oil, relative to price index for bus. sector output"
        @dlog(poilr[t]) - poilr_a[t] = y_poilr[1] * log(poilr[t - 1] / poilrt[t - 1]) + y_poilr[2] + y_poilr[3] * @d(log(poilr[t - 1]), 0, 1) + y_poilr[4] * @d(log(poilrt[t]), 0, 1)
        "10-year expected PCE price inflation (Survey of Professional Forecasters)"
        ptr[t] - ptr_a[t] = y_ptr[1] * ptr[t - 1] + y_ptr[2] * picxfe[t - 1] + y_ptr[3] * pitarg[t - 1]
        "Price of adjusted final sales excluding consumption"
        @d(log(pxnc[t]), 0, 1) - pxnc_a[t] = pipxnc[t] / 400
        "Price index for final sales plus imports less gov. labor"
        log(pxp[t]) - pxp_a[t] = log(pxp[t - 1]) + y_pxp[1] * log(pcnia[t] / pcnia[t - 1]) + y_pxp[2] * log(pxnc[t] / pxnc[t - 1])
        "Price index for exports, cw (relative to PXP)"
        (log(pxr[t]) - pxr_a[t]) - log(pxr[t - 1]) = (y_pxr[1] + pipxnc[t] / 400 + dpadj[t]) - @d(log(pxp[t]), 0, 1)
        "Desired Inventory Sales Ratio"
        log(qkir[t]) - qkir_a[t] = (1 - dglprd[t]) * y_qkir[1] + log(qkir[t - 1])
        "Random walk component of non-oil import prices"
        log(qpmo[t]) - qpmo_a[t] = log(qpmo[t - 1]) + y_qpmo[1]
        "S&P BBB corporate bond rate, risk/term premium"
        rbbbp[t] - rbbbp_a[t] = y_rbbbp[1] + y_rbbbp[2] * zgap10[t] + y_rbbbp[3] * ((rbbbp[t - 1] - y_rbbbp[1]) - y_rbbbp[2] * zgap10[t - 1])
        "New car loan rate at finance companies"
        rcar[t] - rcar_a[t] = y_rcar[1] + y_rcar[2] * d79a[t] + y_rcar[3] * ((1 - d79a[t]) * t47[t]) + y_rcar[4] * rcar[t - 1] + y_rcar[5] * rg5[t] + y_rcar[6] * rg5[t - 1]
        "Rate of capital gain on the non-equity portion of household wealth"
        rcgain[t] - rcgain_a[t] = picx4[t] + y_rcgain[1] + y_rcgain[2] * xgap2[t] + y_rcgain[3] * (((rcgain[t - 1] - picx4[t - 1]) - y_rcgain[4]) - y_rcgain[5] * xgap2[t])
        "Real expected rate of return on equity, premium component"
        reqp[t] - reqp_a[t] = y_reqp[1] + y_reqp[2] * rbbbp[t] + y_reqp[3] * ((reqp[t - 1] - y_reqp[1]) - y_reqp[2] * rbbbp[t - 1])
        "Federal funds rate"
        rff[t] - rff_a[t] = (1 - dmptrsh[t]) * max(rffrule[t], rffmin[t]) + dmptrsh[t] * max(dmptr[t - 1] * rffrule[t] + (1 - dmptr[t - 1]) * rffmin[t], rffmin[t])
        "Value of eff. federal funds rate given by estimated policy rule"
        rffalt[t] - rffalt_a[t] = y_rffalt[1] + y_rffalt[2] * rff[t - 1] + y_rffalt[3] * rff[t - 2] + y_rffalt[4] * xgap2[t] + y_rffalt[5] * xgap2[t - 1] + y_rffalt[6] * ((picxfe[t] + picxfe[t - 1] + picxfe[t - 2] + picxfe[t - 3]) / 4)
        "Value of eff. federal funds rate given by the inertial Taylor rule"
        rffintay[t] - rffintay_a[t] = y_rffintay[3] * rff[t - 1] + (1 - y_rffintay[3]) * (rstar[t] + (picxfe[t] + picxfe[t - 1] + picxfe[t - 2] + picxfe[t - 3]) / 4 + y_rffintay[1] * ((picxfe[t] + picxfe[t - 1] + picxfe[t - 2] + picxfe[t - 3]) / 4 - pitarg[t]) + y_rffintay[2] * xgap2[t])
        "Federal funds rate"
        rffrule[t] - rffrule_a[t] = dmpex[t] * rfffix[t] + dmprr[t] * (rrfix[t] + (picxfe[t] + picxfe[t - 1] + picxfe[t - 2] + picxfe[t - 3]) / 4) + dmptay[t] * rfftay[t] + dmptlr[t] * rfftlr[t] + dmpintay[t] * rffintay[t] + dmpalt[t] * rffalt[t]
        "Value of eff. federal funds rate given by the Taylor rule with output gap"
        rfftay[t] - rfftay_a[t] = rstar[t] + (picxfe[t] + picxfe[t - 1] + picxfe[t - 2] + picxfe[t - 3]) / 4 + y_rfftay[1] * ((picxfe[t] + picxfe[t - 1] + picxfe[t - 2] + picxfe[t - 3]) / 4 - pitarg[t]) + y_rfftay[2] * xgap2[t]
        "Value of eff. federal funds rate given by the Taylor rule with unemployment gap"
        rfftlr[t] - rfftlr_a[t] = rstar[t] + y_rfftlr[1] * pitarg[t] + y_rfftlr[2] * (picxfe[t] + picxfe[t - 1] + picxfe[t - 2] + picxfe[t - 3]) + y_rfftlr[3] * ((lurnat[t] + deuc[t] * leuc[t]) - lur[t])
        "Average yield earned on gross claims of US residents on the rest of the world"
        @d(rfynic[t]) - rfynic_a[t] = y_rfynic[1] + y_rfynic[2] * (rfynic[t - 1] - rfynil[t - 1]) + y_rfynic[3] * @d(rfynic[t - 1]) + y_rfynic[4] * @d(rfynil[t])
        "Average yield earned on liabilities of US residents on the rest of the world"
        @d(rfynil[t]) - rfynil_a[t] = y_rfynil[1] + y_rfynil[2] * rfynil[t - 1] + y_rfynil[3] * rg10[t - 1] + y_rfynil[4] * rtb[t - 1] + y_rfynil[5] * reqp[t - 1] + y_rfynil[6] * @d(rfynil[t - 1]) + y_rfynil[7] * @d(rg10[t]) + y_rfynil[8] * @d(rtb[t]) + y_rfynil[9] * @d(reqp[t])
        "10-year Treasury bond rate, term premium"
        rg10p[t] - rg10p_a[t] = y_rg10p[1] + y_rg10p[2] * zgap10[t] + y_rg10p[3] * d8095[t] + y_rg10p[4] * (((rg10p[t - 1] - y_rg10p[1]) - y_rg10p[2] * zgap10[t - 1]) - y_rg10p[3] * d8095[t - 1])
        "30-year Treasury bond rate, term premium"
        rg30p[t] - rg30p_a[t] = y_rg30p[1] + y_rg30p[2] * zgap30[t] + y_rg30p[3] * d8095[t] + y_rg30p[4] * (((rg30p[t - 1] - y_rg30p[1]) - y_rg30p[2] * zgap30[t - 1]) - y_rg30p[3] * d8095[t - 1])
        "5-year Treasury note rate. term premium"
        rg5p[t] - rg5p_a[t] = y_rg5p[1] + y_rg5p[2] * zgap05[t] + y_rg5p[3] * ((rg5p[t - 1] - y_rg5p[1]) - y_rg5p[2] * zgap05[t - 1])
        "Average rate of interest on existing federal debt"
        rgfint[t] - rgfint_a[t] = ((y_rgfint[1] * rgfint[t - 1] + 0.14 * rgw[t - 1]) * gfdbtn[t - 2]) / gfdbtn[t - 1] + rgw[t - 1] * (1 - gfdbtn[t - 2] / gfdbtn[t - 1]) + y_rgfint[2]
        "Interest rate on conventional mortgages"
        @d(rme[t], 0, 1) - rme_a[t] = y_rme[1] + y_rme[2] * @d(rg10[t], 0, 1) + y_rme[3] * d87[t] * @d(rg10[t], 0, 1) + y_rme[4] * (rg10[t - 1] - rme[t - 1]) + y_rme[5] * d87[t] * (rg10[t - 1] - rme[t - 1])
        "Expected long-run real federal funds rate"
        rrtr[t] - rrtr_a[t] = y_rrtr[1] * rrtr[t - 1] + y_rrtr[2] * rrff[t]
        "Equilibrium real federal funds rate (for monetary policy reaction functions)"
        rstar[t] - rstar_a[t] = rstar[t - 1] + y_rstar[1] * ((rrff[t] - rstar[t - 1]) * drstar[t])
        "Average government corporate income tax rate"
        trci[t] - trci_a[t] = trcit[t] + y_trci[2] * xgap2[t] + y_trci[1] * ((trci[t - 1] - trcit[t - 1]) - y_trci[2] * xgap2[t - 1])
        "Average government tax rate for personal income tax and non-tax receipts"
        trp[t] - trp_a[t] = y_trp[1] * trpt[t] + y_trp[2] * (trp[t - 1] - trpt[t - 1]) + y_trp[3] * (trp[t - 2] - trpt[t - 2]) + y_trp[4] * xgap2[t]
        "Average government tax rate for personal income tax, trend"
        trpt[t] - trpt_a[t] = dfpex[t] * trptx[t] + dfpdbt[t] * (trpt[t - 1] + y_trpt[1] * (gfdbtnp[t - 1] / xgdpn[t - 1] - gfdrt[t - 1]) + y_trpt[2] * @d(gfdbtnp[t - 1] / xgdpn[t - 1] - gfdrt[t - 1], 0, 1)) + dfpsrp[t] * (trpt[t - 1] + y_trpt[3] * (gfsrpn[t - 1] / xgdpn[t - 1] - (gfsrt[t - 1] + 0.0075 * xgap2[t - 1])))
        "Federal Government budget surplus, residual"
        ugfsrp[t] - ugfsrp_a[t] = y_ugfsrp[1] * (1 - y_ugfsrp[2]) + y_ugfsrp[2] * ugfsrp[t - 1]
        "Multiplicative factor for government civilian employment"
        log(uleg[t]) - uleg_a[t] = log(uleg[t - 1]) - 0.1 * (leg[t - 1] / lep[t - 1] - adjlegrt[t])
        "Stochastic component of trend ratio of PCNIA to PXP"
        log(uqpct[t]) - uqpct_a[t] = y_uqpct[1] + log(uqpct[t - 1]) + huqpct[t]
        "Stochastic component of trend ratio of XGDPT to XBT"
        log(uxbt[t]) - uxbt_a[t] = y_uxbt[1] + log(uxbt[t - 1]) + 0.0025 * huxb[t]
        "Corporate profits, residual"
        uynicpnr[t] - uynicpnr_a[t] = y_uynicpnr[1] * (1 - y_uynicpnr[2]) + y_uynicpnr[2] * uynicpnr[t - 1]
        "Desired investment-output ratio"
        vbfi[t] - vbfi_a[t] = (uvbfi[t] * (pkbfir[t] / pbfir[t])) / rtbfi[t]
        "Residual Factor (Potential business output)"
        log(xbtr[t]) - xbtr_a[t] = y_xbtr[1] * log(xbtr[t - 1])
        "Final sales of gross domestic product, cw 2012\$"
        log(xfs[t]) - xfs_a[t] = log(xfs[t - 1]) + y_xfs[1] * log(ecnia[t] / ecnia[t - 1]) + y_xfs[2] * log(eh[t] / eh[t - 1]) + y_xfs[3] * log(ebfi[t] / ebfi[t - 1]) + y_xfs[4] * log(egfe[t] / egfe[t - 1]) + y_xfs[5] * log(egfl[t] / egfl[t - 1]) + y_xfs[6] * log(egse[t] / egse[t - 1]) + y_xfs[7] * log(egsl[t] / egsl[t - 1]) + y_xfs[8] * log(ex[t] / ex[t - 1]) + y_xfs[9] * log(emo[t] / emo[t - 1]) + y_xfs[10] * log(emp[t] / emp[t - 1])
        "GDP, cw 2012\$"
        log(xgdp[t]) - xgdp_a[t] = log(xgdp[t - 1]) + y_xgdp[1] * log(xfs[t] / xfs[t - 1]) + y_xgdp[2] * (log(ki[t]) - log(ki[t - 1])) + y_xgdp[3] * (log(ki[t - 1]) - log(ki[t - 2]))
        "Final sales plus imports less government labor, cw 2012\$"
        log(xp[t]) - xp_a[t] = log(xp[t - 1]) + y_xp[1] * log(ecnia[t] / ecnia[t - 1]) + y_xp[2] * log(eh[t] / eh[t - 1]) + y_xp[3] * log(ebfi[t] / ebfi[t - 1]) + y_xp[4] * log(egfe[t] / egfe[t - 1]) + y_xp[5] * log(egse[t] / egse[t - 1]) + y_xp[6] * log(ex[t] / ex[t - 1])
        "Imputed income of the stock of consumer durables, 2012\$"
        log(yhpcd[t]) - yhpcd_a[t] = log(y_yhpcd[1]) + log(kcd[t - 1])
        "Dividends (national income component)"
        @dlog((ynidn[t] - ymsdn[t]) / pxb[t]) - ynidn_a[t] = y_ynidn[1] * log(qynidn[t - 1] / (ynidn[t - 1] - ymsdn[t - 1])) + y_ynidn[2] * @dlog((ynidn[t - 1] - ymsdn[t - 1]) / pxb[t - 1]) + y_ynidn[3] * zynid[t]
        "Net interest, rental and proprietors' incomes (national income components)"
        ynirn[t] / xgdpn[t] - ynirn_a[t] = y_ynirn[1] + y_ynirn[2] * (ynirn[t - 1] / xgdpn[t - 1]) + y_ynirn[3] * 0.01 * @d(rbbb[t])
        "Expected trend share of property income in household income"
        log(zyhpst[t]) - zyhpst_a[t] = log(zyhpst[t - 1]) + (y_zyhpst[1] * yhpgap[t - 1]) / 100
        "Expected trend ratio of household income to GDP"
        log(zyhst[t]) - zyhst_a[t] = log(zyhst[t - 1]) + (y_zyhst[1] * yhgap[t - 1]) / 100
        "Expected trend share of transfer income in household income"
        log(zyhtst[t]) - zyhtst_a[t] = log(zyhtst[t - 1]) + (y_zyhtst[1] * yhtgap[t - 1]) / 100
    end

    return @initialize m
end
