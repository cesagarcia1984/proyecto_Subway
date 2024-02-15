import pandas as pd
import requests
import re
from functools import reduce
import json 
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

def leer_archivo(f_path):
    """
    Lee el archivo según su extensión. Disparado por la funcion "captura_evento"
    Args:
        event (dict): Event payload.
        file_path (str): ruta del archivo
        file_type (str): tipo del archivo
    """
    # Extraer el tipo de archivo
    f_type = f_path.split('.')[-1]
    
    # Revisando si archivo es csv
    if f_type == 'csv':
        # Leyendo archivo en dataframe
        df = pd.read_csv(f_path)

    # Revisando si archivo es json    
    elif f_type == 'json':
        try:
            # Intentar leer el archivo json como si no tuviera saltos de linea
            df = pd.read_json(f_path)
        except ValueError as e:
            if 'Trailing data' in str(e):
                # Leer el archivo json conteniendo saltos de linea
                df = pd.read_json(f_path, lines = True,  dtype={'user_id': str})
            else:
                # Cualquier otro error
                print('Ocurrió un error cargando el archivo JSON:', e)

    # Revisar si el archivo es tipo parquet
    elif f_type == 'parquet':
        # Leyendo archivo en dataframe
        df = pd.read_parquet(f_path)

    # Revisar si el archivo es tipo pkl (Pickle)
    elif f_type == 'pkl':
        try:
            # Leyendo archivo en DataFrame desde Google Cloud Storage
            df = pd.read_pickle(f_path)
        except Exception as e:
            print(f'Ocurrió un error al leer el archivo Pickle: {e}')
    
    return df

def limpiar_df(df):
    """
    Limpia el df "sales_count_month". Disparado por la funcion "captura_evento"
    Args:
        data (DataFrame): dataframe a limpiar.
    """
    
    try:
        #================= ETL ================================
        lista_locales_newyork = ['0x89c2e7f1c5ffad57:0x188085b588f042e3', '0x89c27d7e84956c31:0x301d481a1d010206', '0x89d409252d58e323:0xdce89aff887ec3e5', '0x89dd6b23f8e55e27:0x19811380ec6fb85b', '0x89d941bdc690b91f:0x11a098273ba3e0a', '0x89da6a4d33585de9:0xb367d704bfa506cd', '0x89de72120cdb656b:0x621820900f61892f', '0x89dea7b7a95bb73f:0xb6e327404c1c645e', '0x89d04caf4303d843:0x925d32f0be6d1a7d', '0x89c2f6642a4ee3d7:0x8129f905ad56e6a6', '0x4ccd05bfcefa8ebd:0xb871d9ce9cf78b62', '0x89e9d4fe087142ab:0x6710489edd01f7c5', '0x89c281ae0a5e7673:0x62be81fc97428d42', '0x89c2877953318039:0xb59100dd5346f314', '0x89d39988b4aa672f:0xdf91d6ce7746d590', '0x89dd3047a9a381d9:0xf6a79137095b45b6', '0x89c287860e441db5:0x6502bb32bf555029', '0x89c25e495c5cb657:0x5b6cd00a3ce09de5', '0x89c2e85224873171:0x2bab27118a215c28', '0x89d06439e2fb272b:0x3824fa15186f20c6', '0x89d9f224f3834c8f:0xb16fab0164e6b9b4', '0x89c264be18b26929:0x504b36865da86670', '0x4ccc157d78c0a665:0xea9018372b64f529', '0x89c259a63a91f1dd:0xfcd9a4cf6f27dd00', '0x89c2645571634cad:0xf26b172bd1b546c5', '0x89c259ac7facb707:0xb3c6cd387252fbcf', '0x89c2c71880b409d9:0xe4cae23461dbd3bf', '0x89dd47651d5dfac9:0xe384c344a50affb5', '0x89c2457db1e36ded:0x797945a69d40c21e', '0x89d30aea5737bc19:0xb6722f735c102e45', '0x89d976de5cd2c12d:0xb1a010b6e284ab4f', '0x89c244f4e0d95e45:0x5633520c792a6340', '0x89c29748c32ca5c9:0x2d3414fa86aebd2c', '0x89c27e1c794ee6f5:0x4132fe37dada9bde', '0x89d9f22c44b72057:0xb2284ffe32c17f6f', '0x89c25ed9414a21d9:0x5c27e48471e1fd27', '0x89e83bad1f6a1927:0x9a4b72eb10cbed89', '0x89d77d6ad1b07f0f:0x14196fede1a1f271', '0x89c2f3bf1dbfe66b:0x6d2661fddcd5070d', '0x89de702bb4828461:0x236ed90ec56891a9', '0x89d10fe992aaee71:0xde917cd8de721c75', '0x89c2ba6e00cb2295:0xf13e89db6da5f882', '0x89c2599cd89b9171:0xac09a0542a8d7bc4', '0x89c28dbbb7d536b7:0xf4c42ee1c6d54d61', '0x89c3b5596d1ef5d7:0x68ad9a64d43e7957', '0x89c28d90ee432931:0xe96a71818b30f22', '0x89c2f3a56de23b5b:0x7257807ed11b2e82', '0x89d174894e27cb67:0x39d7a9a661734b25', '0x89c25ac104e32733:0x16a707e52a81fb6c', '0x89d2cb8180844b5f:0xcdf3804458ae9ff0', '0x89d2f8f854f41735:0x2e66b86527a654b2', '0x89c25c22323a789d:0xe33baad738fcdca5', '0x89e848cdaff54cc7:0xac0eb634fe553f4b', '0x89c25bb181cb01eb:0x52d25787166cf750', '0x89d8654a696e9077:0x1e5cfc21ff10b74e', '0x89d27dbe60653767:0x489293820c29b657', '0x89c27d5e99f9702f:0x14f3cfb70b5f92bb', '0x89d167a64b254907:0x43608400ca0cc078', '0x89c25b6407f98951:0xf66b73b4a091bbd0', '0x89c24fcd4703724b:0x86629101b17410f9', '0x89c2455bb8e95d9b:0x44ce98b5352a4c59', '0x89c27b3a371ec279:0x1f6681b5788756be', '0x89c25c268c8125ed:0x2199e7c6bc771072', '0x89c2588a79a5618b:0x232b3f892b8c48bf', '0x89c2632a39b67af7:0x2c87ce72399d2e31', '0x89c28a0aababe6c5:0x9d7fd9beadd6a346', '0x89c28106c09d9ae3:0x4797caaf7d10ace5', '0x89dc6be714a1a8e1:0xec821ac4d302e5d0', '0x89c28ed762108615:0x53e3e5d7961cb426', '0x89e8319177a9eb1d:0x8fdf2f4004beaf', '0x89d3726995c65d4d:0xab4cb0f7d8e22829', '0x89d3967484834721:0xb095c38d3c6249e3', '0x89d27084bc2605f1:0x54ac7ebc4216f3a8', '0x89e82d00ef04e321:0x342ab1a45612cc0e', '0x89e82e31e5b68d11:0x50f0dfbf5fcb2e88', '0x89c262aad51423e7:0x311b2f2b120132a7', '0x89c2603e83d2da8f:0x82ef66db4015a52c', '0x89de3720f07e9b63:0xcc2d233a782659d3', '0x89e82d3bee39efc9:0x94816263523a6aee', '0x89c2dd506e9ed7b3:0x87d632f138219be8', '0x89c32d19dc051377:0xb9ce7b3b7ebdb757', '0x89d3edde3fcf22d5:0x7797160f8e5b77a7', '0x89de782bca1f3113:0xea40eefb024acfb5', '0x89c261204c34d8e9:0xf8b5b1230547402a', '0x89d6b132398d47b9:0x799eff04d901e574', '0x89dfd1281e7e4137:0x878917bb96dab1a5', '0x89c25a2a4ed6af69:0x1ebf80fb04dce88a', '0x89d18c45dc5667c9:0x3a8c35c92d72cd0c', '0x89c24bad8f47f515:0x2245714c9ccc52d4', '0x89c27f1d066fb207:0xd85604a4ae1cbbcd', '0x89dc2ef34df02e3f:0x743977fbd9148da4', '0x89c25985aed61a37:0x69555d0213d71cd8', '0x89c2975c986ca4bb:0x73ea92f2435a4a4e', '0x89de22756d622cf3:0x63109760d25343eb', '0x89d0add738fb042f:0xac754a20e333915e', '0x89c28806b6183eb7:0xec16b282a0351067', '0x89c2811951f4ac7d:0x4989fa83367d5ded', '0x89c2ba3eb103b69d:0x547e105ccb59046c', '0x89c27e22196edf0f:0x86817a6fad97f28a', '0x89c2624567663a19:0xeb8b3d44f3820469', '0x89de0b212beb06d1:0x181a6b8facd071c4', '0x89d98cdc8f0d62db:0x2b8dcc014c75cf77', '0x89d08558946f7799:0xd77ae8b465a1f9e4', '0x89c258508ea1a69b:0x169c0d62b6e30507', '0x89e82b3cfbb5a527:0x49facbc3c251e6f2', '0x89c2599434aba5cf:0x6dbf07ac48f57bf8', '0x89c25f1da44bb69d:0x942112dcc65c4c61', '0x89c2f60a65d3cb5d:0xba70539e0c045b30', '0x89d9c870b39df96f:0x32e609ab808ceefe', '0x89d3a28fdd6d2945:0x498affbe1c1cb892', '0x89d30d5a82e96393:0x957ed28f829e4700', '0x89de0ddcd971712f:0xc89a7b270b6e6632', '0x89d1b2942a752c0b:0x4beb973aa839bca0', '0x89e83807f9672919:0xb2dd4e69931a2f65', '0x89e08354a8973c49:0x4dcc95e38ad33471', '0x4cca29c78eae54b5:0xc3f2ac97edba4fd8', '0x89dda740e2825b79:0x68365ca2cb96a848', '0x89d6b736898b4287:0xb5ac55cd1c038d2f', '0x89d9f0b7bda4704b:0x8eb004aa15e9286d', '0x89e838efd271bb97:0x731787a29fa1b884', '0x89c2e968d79f9eb9:0xc512e28735cf07d', '0x89d0152dce89f14b:0xfe783d0f991ae480', '0x89c25a3d425eb491:0xefaf04f01e414f47', '0x89d21db735e80881:0x6c34f5ea48c4f93c', '0x89c34f549f2de2cd:0x5d7beb5c122add58', '0x89c259001db887bd:0xa5f95d7eab5909d1', '0x89d04833fbd92bc7:0x53c5853068f6cea1', '0x89e82fec19f3caa9:0xff8af23690dddfa', '0x89dd3164c8a42885:0xe6c07d73d761fe21', '0x89c25b03b4e01579:0xaa513df1b39837b5', '0x89dd26bc4bcf8ae1:0xa887f205789c5a20', '0x89daeede8fb0908d:0x90205a5d20eddf53', '0x89dd29ae2a18c6b5:0x93b98fa2ee17acb8', '0x89ddb9bebdefc875:0xefc81841774fea4d', '0x89d30f702e79c1c1:0xc77602f2a0620434', '0x89c259844ee97d25:0x2dd0f46d9ed2f338', '0x89c2e9e643e7f0cd:0x5f3f0de56b64c0a1', '0x89dde00c39fe0073:0xb4ee6ef98786ef36', '0x89c2f4a4e522bc39:0x160ad337ad4e312', '0x89d87bd7aa9b6e5b:0x963ccf7f3259a2c0', '0x89d31bdd32f0144b:0x23e99a214eaebeb2', '0x89d9df9f230027bd:0x2ac0db8b418e164a', '0x89c266d59e9fa391:0x71e0312005d016ac', '0x89c2f5ba72a356dd:0xd12989ddf2c1404a', '0x89e830a4abb39897:0xd7212e46dea8a649', '0x89c24e24415b8c77:0x1ed94f398c271977', '0x89d4095b9ae24733:0x371bb7e4a31ee092', '0x89d31da8c534f1eb:0xfa144392ea73fbb2', '0x89c2f6880b8f9fe1:0xc29d17a454962a9b', '0x89dfd21c8acc988d:0x9643d9170334e04d', '0x89c27b7f52eb67c5:0xc3395b362f6046ea', '0x89c262700603c971:0x7f7ad8a512da231', '0x89c25f3432fcd4d9:0xa31c23f1f63e34c2', '0x89c2584d75b681db:0xe64ed1a40376a9a5', '0x89e82b51b3082e21:0xe179f894ec8e833f', '0x89d9f1e602740ce3:0x8f5e791058c1fe26', '0x89c2f385eaa74315:0x494f6f6ddfab4325', '0x89de0543753534ef:0x3aebfe3bea1480ec', '0x89c2f4c199038171:0xec1ddcbbe9bb1a74', '0x89de0fa8255139fd:0xf5c9ddd703ea2974', '0x89daec8c34b560b9:0x81f0849123d00ab3', '0x89e83a21e2e52a85:0x71b7a075fdb7ed3a', '0x89c2f396bf98b675:0x623a7ba64cab5866', '0x89d6b34e864c3419:0xbe4a9d2749d806c8', '0x89e662e50a8d5c47:0x63a96731fcb63f94', '0x89dafaad89c77f87:0xd583a07a1f54882d', '0x89dd5a742fd22181:0x28a13f60bbf3ab4a', '0x89c27ba076f98ff5:0xf02d550133e535fe', '0x89c261204c34d8e9:0x7a81ce72dd09bc6f', '0x89d8dfc862bcda93:0xe6b90535b933bca7', '0x89d2cc6d2ce1cf67:0xd2c41dfe910f362c', '0x89c27b77158550d9:0xa9924957b60ee4d7', '0x89c25997e0716a7b:0xb15f4582f9b3f104', '0x89c2f6867c770811:0x3abc2d677eb0abf5', '0x89dda740e2825b79:0xde3d47181515e8e7', '0x89c25ef69e53cafd:0xb03d93cf0139a556', '0x89d3091e3aa64fbd:0x4e9f82bc6a85146b', '0x89c2eeaf0c595583:0x1be0427860576834', '0x89c25955c95a5565:0xb911dec1e99f177e', '0x89d3ede6a4092ab7:0x2bdffc52306b9ae6', '0x89c24500776c9a5f:0x2c06c31afff535dc', '0x89dc791c28061ae9:0x37aa256aadb20f0e', '0x89c258fed6d19b35:0xc7c519a5a706e9c4', '0x89e00bc9caa359b9:0xa99b6d5ce4fe8f52', '0x89c25980fcb9e4e9:0x13108c81bbb9582c', '0x89c259a5445381d9:0x2d34f1f980ec7831', '0x89c2f27bc9f7017b:0xb4255b26e645901', '0x89c2616d3fd72809:0x636fc3153dd92ce9', '0x89c2636c5a303d25:0xfb8a94dee8ace0d1', '0x89d8b54d4295c721:0x6158c79886490e25', '0x89deb68bd0063e11:0x2c51c47357ffe9c2', '0x89c245b8bfceac8d:0x64d21006eff2db7a', '0x89d6b0fe1ee77539:0xe0e71d9d69ab86f6', '0x89c25e4d80195625:0x8acc8e4c04a98f3b', '0x89c2494fa4795f65:0x6288e72ced272692', '0x89ddc60fe2fcd0a1:0xe6738b0d075f19b9', '0x89e832446c7159bf:0x782c7faa70da9544', '0x89c2f5d6acc80945:0xc3d004db755665ee', '0x89c25ad2387709cb:0x5cc69cfdcba31f67', '0x89c262289e6a0025:0x15070df23765bfc1', '0x89c25f67eb74b71d:0x516c4ceaf6360c68', '0x89c2f49f04c8062f:0xd7b92c104f40122b', '0x89c25f3b87e48cbf:0x88fd2a0ce584330d', '0x89d306063ec81489:0xa5b4720cf55cd018', '0x89dcd13f9ba19b77:0xb10f8ad9eac35dcd', '0x89d375cde8d180df:0x909ad98f26bbaf1f', '0x89c2f4721ec32233:0x976b45379671d2d0', '0x89d77cb46f8fd0b3:0xaebda47f35c7c7e', '0x89de5f6046a0a3ef:0xffc137eedc02f92a', '0x89e8290f319da065:0x24edf9cf02b1e742', '0x89c2943a22d511e3:0xe233d6db0c80543f', '0x89c2f25c490e0097:0x469c25f6b9613c4f', '0x89c29517b1840191:0x58ec8dcbd075d917', '0x89d9f8961bed3541:0x3800b499c2d57bba', '0x89c28a873a7fb901:0x702cf79eddadac77', '0x89dfcf82a321469f:0x6784b3010c91aa98', '0x89c258bcfef05589:0x88b94b984049ab86', '0x89c2f60ee2bd441b:0xb34e86e02a0a2a6c', '0x89e833deeb37d42f:0xb97d2063c281a32a', '0x89c289aea234ac79:0x55216897e4552220', '0x89d8707677781cf5:0xbd9da674c4609155', '0x89d0d32a6107d2c5:0x1491bab8b5765f1b', '0x89d9e4f7f1f831e5:0x40b7e182e5baa8c1', '0x89c26003eb549baf:0x54c133c2298e1b0', '0x89d144b01c5f5069:0x3922ee57a0f97c9', '0x89c2599434aba5cf:0x342e9f41cb3e9bce', '0x89c27b42978c553d:0x22c5a6df20333aff', '0x89c2646976b47fe9:0x105298f5c40139da', '0x89c2ed7a2ec28365:0x35c609f093ac8cb1', '0x89c25f99477a1abf:0x7cb97313032d9a07', '0x89c32d6f11edb09d:0xcd22538ac05f4c86', '0x89e8490e5e03e5a1:0x2810b0b0a598fa8b', '0x89d9ef03e7c63b5b:0xbff28dec7ec55f17', '0x89c3326ec7b9ddc9:0x73dbb9c13524269f', '0x89d6ac48e8e89b8d:0xa95c62315a18d691', '0x89e82dc54ef322cf:0xbf20e00fca658f23', '0x89c26100ec2d6e33:0x798698ed2a0bdd68', '0x89db90698fbdd7e7:0x71001bd99e91337', '0x89c2c3e1a612e687:0xbb28703fca644123', '0x89c2f66a18f452a7:0x7e640f73cac75042', '0x89c28aede84618f5:0x51d69cfffe9c803', '0x89c2822f70891c21:0x519242a0ea42f2ba', '0x89c24faf0faa31a5:0xb4201ca37deeaa7', '0x89c258541b904dbb:0x8fe6de46ba0c386c', '0x89e825f9341b7441:0x1c3f9b20397ea539', '0x89d6d753b35f9565:0x6890d00a5928e413', '0x89d6d40cd8cccd7f:0xac0bc6ad4f69b577', '0x89d6ccf132c32f51:0xdb78bb8ae1053511', '0x89c2e8917ad38cd5:0x284d9c02dd909246', '0x89e83a9ff11e4061:0x8abbd82b9c1bd10c', '0x89de6045405a4ad1:0xe41514dac56f88d0', '0x89d36dc9ccad975b:0x52d0f49546b401f5', '0x89c244a69d6bc6a5:0x529c127d37a52732', '0x89d9df82a929903f:0x77de117b49c1ae4c', '0x89d3727b53b21cf3:0x39497cdedb0430fe', '0x89dfc10b69ba6363:0xc50438744fab3732', '0x89d0401d6ddb3821:0x8ed2ed04a389d4a1', '0x89daec4bd267453b:0x22a5f058ff42f5ea', '0x89de0b3f03e435e5:0x524aae747d713518', '0x89d378d72a0dfe25:0xb98f909c08d7175a', '0x89c25a163d229f4b:0x961287df9969a62', '0x89c25ab346526f65:0x84adee98d631cb1a', '0x89c265dadbf22b4b:0xcae438c7b6a60133', '0x89d36f5573aed2ef:0x8682e3ae850d0cdd', '0x89c2f43eb118c741:0xd1837034d36dd75f', '0x89d9f45d253b9733:0xa6a86b437fd2e529', '0x89e842885271bfdf:0x70f6189c4c15600e', '0x89c2f367b9732689:0xeb92ff91fe085c08', '0x89c2919b59d5e4e9:0xf035846282020de7', '0x89de71913d00b84d:0x270983995548cf', '0x89da0ae5e9915299:0x935551305d28be5', '0x89dfd1e64a544617:0x4349b6d5f1fac5d2', '0x89d12ac59ed3ef03:0x97044dda1e9b7e07', '0x89c25fab9fab38f3:0x4995c99616a1e57a', '0x89ddeae426065041:0x91fb9ae63f7e4002', '0x89d371c56e90338f:0xbbe986289f86b6e2', '0x89c258b8c9edf427:0x288c6931b057fe09', '0x89d30ef0169ad621:0xd10bf519efeef933', '0x89dd2df87c65aebb:0xa7e028cb14ba057c', '0x89e8677cffd259f5:0xf5fce35e5919cdd6', '0x89d9f31097803393:0xbdb92eca7d9cb28f', '0x89dfd1afae903069:0xd35fe5c5ca95bc18', '0x89c27d1f6eecfa97:0x41a164890819e0a7', '0x89dd37010e5964df:0x64cdbdf6f165f837', '0x89d97e3e17d9f5a9:0x4f7f01c234ae1730', '0x89e829c7e191f8b7:0x6295af59de1c3999', '0x89d725a58ac9e1c1:0x51e2526144102fbb', '0x89c2f66250659959:0x9ed2ac09ffd97edf', '0x89d2cc6bb84944b5:0xe366c6eb5de68f3d', '0x89e83adf763c4e19:0x78d85f29d70acc19', '0x89c264a975bf93b7:0x69e59550ddadd456', '0x89c2eca9e4d6d4ab:0x3cdcb7e6b6cc2b4f', '0x89c25de6c2a57ad1:0x56eca81a4b235133', '0x89c2f41fac369577:0xbdf5b103152d1d57', '0x89c25e9007fb307f:0xd510cbd2cb9d60c', '0x89c2f2ac9468e91f:0x17e4d0e9de4fce70', '0x89c2617b5c8b1ead:0x385759ca86e266a5', '0x89c2f4c0a81f6153:0x6d53a3f614664f75', '0x89c2f4400875ecbf:0x788f660593f8cdb3', '0x89c259aa6f3f0f4d:0x386491acaa1b3202', '0x89c282714ec476f7:0x668e29bc613ed863', '0x89de8dc36f523fa5:0xff7a8273d011dea5', '0x89c25f255ac80939:0x4bf393bc177963c9', '0x89d0410f3b6e4d99:0xd3cd865c5458d8f5', '0x89d1268d5ad3b179:0x9138d40ec644829a', '0x89d30cbdf468207d:0xb2261ab237e1482a', '0x89dd2d729fe64519:0x4de0a5defbb448bf', '0x89c258fb9d086d91:0xb93bc77e0029923a', '0x89c2811df2a55bb3:0x22c6e5f58a10ca5', '0x89d9f3b379cee7b3:0x65513038b1a0f4cb', '0x89d0b7cb24e067f3:0x3f6c6912dd33252', '0x89c260c09b2d8f1f:0x6b1f9564f2ee562b', '0x89dd39bd1e8c45e3:0xd6eef5f3b1f7eab1', '0x89d0c4d61e66d59b:0xd245220b65874e9b', '0x89c261f1097348db:0x1c66eea6afbb534f', '0x89d312dbe5439da5:0x84ac2a0b69bfaabe', '0x89dfce4946792489:0x60cb0a0af947f6ac', '0x89d37210c02eea37:0x210aadb62d06f21f', '0x89c27dfcd9fdbe61:0xd68f06401f42b64', '0x89c25924973c297f:0x77c69d9a136b0896', '0x89c2f62395535f8d:0x2682575c10a4673f', '0x89d2170589b7ae6d:0xd00ced944e4a6ca', '0x89c25f03bf02cda3:0xe3424b6c0483bbe0', '0x89d3124e83fe18ff:0xe17622cbb3f94087', '0x89d0811c1f963c47:0x32e0ff55c0d926a7', '0x89c2f48d1af38bc3:0x333f473e5c6d8c42', '0x89c259049ac882e5:0x55c81fb781cf4986', '0x89dde1ad6558d285:0x7856158e11e97c80', '0x89c2f4596fce0833:0x8ef18997e48090ab', '0x89e843658c94433b:0xa463ca8478b7b98a', '0x89d996b4ce69a389:0x585cafa7b212dd34', '0x89c25afdb207d4c5:0x55d766c290bc98f', '0x89c27bf4b99cc8d9:0x6687b0e44ac1a664', '0x89c25ae350645f21:0xded7a0bec1030ffc', '0x4ccc674b82f23a79:0x6a049d7202355181', '0x89dde0246c4e12f7:0xe44ecd8bab4ed9f1', '0x89c27db2f217ec5b:0xb8bfaae658a95845', '0x89c24ec45db4fdf5:0xd18fb482a76bb58f', '0x89d9f14dcd76e103:0xf881086a3827e602', '0x89e82cc69d22dc8f:0xe38e48a630b2a45c', '0x89de0c75c9d22927:0x17a8a7a14b0966b6', '0x89c288cc963e0e8b:0x8c26544e4b9e6ae1', '0x89de0a543c43dab1:0x9751f27ff3ca077', '0x89d9f3a1b3373329:0xeb55367bab55c39', '0x89c2598a9a97e465:0xa88cbeb5e5ff5925', '0x89d9f2f185a55765:0xf450f893f93484e4', '0x89d372bdff551595:0xad994c3851cf1409', '0x89c28d22c600c4b9:0xb06e10bff48d364c', '0x89c25edfb76b6d13:0xe67f832318746417', '0x89de13fd86e41615:0x61fc2bf517696b36', '0x89d126fcb49733eb:0x7c0710e26c802f63', '0x89c25c83bd141007:0x16912e3587f4199', '0x89d3e36e543c7ed7:0x54523b2f1520d312', '0x4cca38b9b6829fd5:0xa9fc85f4da8b977d', '0x89d3a7490cbc0877:0x62837e9bd958ce43', '0x89c2f5ca0a544cd9:0xe970ba29c5ec0d03', '0x89c2f38137ac61f1:0x669795524d766985', '0x89dcd2a7a866e329:0x2932753cfdbb95aa', '0x89dd0f24ac98242f:0xf097755a9296a254', '0x89c2803dd9374833:0xcdb113fcf86bfec4', '0x89d9ce5b88594c9f:0xce6bc4aac0868614', '0x89c25a18bdf2ee5d:0x230d21d5b17a273b', '0x89c25ed8fd45455d:0x719b03a496cc4074', '0x89c24e6dd4246685:0x780e2ad66542e939', '0x89c259b779ee8963:0x7fcb3edea44c4a64', '0x89d1af69c9ef3f99:0xca76ddf7804b8f8e', '0x89c25c4f80ecc05d:0x18cd65891733e20d', '0x89c25bade4f8471b:0xccc65e93a848c661', '0x89e82acf59ece4d7:0xe3bf5be735653e8d', '0x89dd2e0bbd3e9b73:0x9a526cd557dffe34', '0x89c25afaaff8caa7:0xf09c705c3ed0c4d', '0x89c2b5c454823ad1:0xad78923089af3b07', '0x89c2f6a06515c02b:0x8213df6f60fa6476', '0x89c2f5d6acc80945:0x52290a98dc4df761', '0x89e846e3be59da8d:0x4b3038a4897f9b0d', '0x89d9e8cba686a8b1:0xed6962e7f5a192f7', '0x89c24dac195d7c2d:0x443b45f11d534011', '0x89d0ab8bc69cd3ab:0x2a35710691fe5c61', '0x89c25908b48c2ae9:0x79033735dc041ba2', '0x89c332e1a365cbf1:0xe93fe5620f1a698a', '0x89de6baffc7dbd37:0xd3e3fb0d5b6657f9', '0x89c25fa7472ef143:0x3c0593848d3cdae1', '0x89c2f61a6be56839:0x75a78d8730967c28', '0x89d943e7f56d8aff:0x1626736a3feee1b4', '0x89de74ce24303747:0x36a9f66ef6919a3c', '0x89e8365298aacd1b:0xbe8574639a51ff40', '0x89c263b7423df215:0x93651787e6ed211c', '0x89d31236366c09db:0x3fe287969c90b3e5', '0x89c2e741fb7d49df:0xcdb25b4d381b280a', '0x89de29a132e338b1:0x1f7d03f648678904', '0x89c259be97932997:0xc8855429ff7cff00', '0x89c2876b67dd4037:0x79d84f546fb152b1', '0x89dd1a4b3f1c95e7:0x5c68cf2b1a765cc7', '0x89c2f5ae9922138f:0xa09e1eaaad46f5a1', '0x89c2f6028c7a548f:0x22b898fef6846b75', '0x89d1529b77ee1399:0xce143cd94011e2de', '0x89c25c1275e5e7b9:0xf3da82b433a177e1', '0x89d39a3f1d331c61:0x8bb4761b218d0523', '0x89c25f993f2d7461:0x57ba9958dcd53492', '0x89d214db8ce3cc3f:0xa970c0bf0f1ca74b', '0x89d971e0b52497d1:0x96178316add46a77', '0x89dc07fa584137ff:0xf821535fac6804e7', '0x89db8b60eaea7b59:0x3b4fd3a4e7d06244', '0x89e9d578648a5d2d:0xd0c3a9cb7f52a76', '0x89c263069d49d8b9:0x99ae8900e5ba445', '0x89e82a53a8b39da5:0x3d9dd0bcf8b36a41', '0x89e9d4aef40e5ec5:0x33a741127258ac2e', '0x89d3752ee020e337:0x1bbb9622d4fe91c8', '0x89d941bbdefe4f77:0x2824351577a7661d', '0x89d94c6aff452493:0x822ed931e0d36855', '0x89d30e1b8fe3bc87:0xa2fea117f30a8252', '0x89d9de4fec147eed:0x8f78339cd1e71252', '0x89d9ec8dc51a9e8f:0x5f95b3492efb548', '0x89c24be3a3998167:0x41cfefba9ccb8f3c', '0x89c24b81b3535037:0x9aae4ca419c70766', '0x89c24e43ffffffff:0x32375c4dc04f0547', '0x89c2f67331abbd67:0xd103714b483a83be', '0x89d941edc9f2e5bd:0x7131af2f84d5209f', '0x89d31b1b5cd7f1f9:0xa1eb7a5afa1e0097', '0x89e832f506582463:0x30a376ff76468565', '0x89c266a33d662571:0xe70ee14f30dba1df', '0x89c25f0bb7485499:0x2280db3fdcf07919', '0x89c25a147fb13335:0xa0da51ec28f5f3ce', '0x89c2f62350e76b7d:0xac835d65e05dc98', '0x89c2598d4765d9d7:0xcdfb4a85fa81b6d0', '0x89d3d992de84e1df:0x3792f494cf7e691f', '0x89c2590a36fbd81f:0x9ace2fe7adc22ef0', '0x89d373d1ad84b7fd:0x41c948c3ad1f7566', '0x89dfd3df6881b763:0x6579d9e7103feb05', '0x89d987a0096e696f:0x2ee8e91c2e3b156e', '0x89d06a758abdcabb:0xba16821496072e76', '0x89e85ea6ce993b4d:0xe7d55ec1300030f8', '0x89c286d4d87a361d:0xe9a149c3915e1807', '0x89c2f671d355bdab:0x7c694abb447f1674', '0x89c2677fac8a827f:0x754e8c196add0f83', '0x89c24ba3cd91488f:0x7314a9664236242d', '0x89c259b1cdb121d1:0x3577ef6e2abf2763', '0x89d082597b76a3d9:0x858132873b203065', '0x4cca4a64ae40751d:0xc3cf2dc1bcbc2495', '0x89c28cba0c7fdcab:0x4cf82096d1c2b4be', '0x89d841c75c9d0d5d:0x56b0c082c4746c6f', '0x4ccb1b147df73e33:0x47ef2e6ee270b62a', '0x89c2f4f44d277f7f:0x29b3c4456af7a7bc', '0x89d95998fe011bd9:0x96a6fea4c88668df', '0x89c2f61699c05ea5:0xf8035b1f42493092', '0x89e88aa7ecc2dd11:0x6a7e750628a42529', '0x89de0f01fbcf2f21:0xe57f6c86bbd2c069', '0x89de04adaadb22b3:0x806c94d485b8a464', '0x89c2805794d143ef:0x8311a0bafb7867cd', '0x89c268ae66d2d48d:0xdc40e38bbb6d05a8', '0x89c258d5992f1dc9:0x64d5601dec95f792', '0x89de05675915ce01:0x6fe536def1002ca6', '0x89d39172bc26cacd:0xdbb26103fa04c671', '0x89d9f3b81592c819:0x3f9926962dcc8213', '0x89d36de9a2496299:0xc28221d36a5721a2', '0x89c2598f87872553:0xbca758c973921236', '0x89da6c431bca8311:0xe3e9e76f9790445e', '0x89c2f5c8bfc7b54b:0xef88d7dde663498b', '0x89c2f6277d12ac13:0x5f167d83bed3892d', '0x89d37656d60bb46d:0x90053545080658b5', '0x89c2f407a1eddf31:0x37895be08863875b', '0x89c245512489af95:0x3cbdcf2050a3cc37', '0x89d47607b56c95bb:0xf702cad49f38e02f', '0x89de17b770c2d66f:0x9ac5bff07baf01db', '0x89c26177165065d9:0xcc493fe80fae495f', '0x89c2599a07af4f13:0x416c9bf5524079e2', '0x89c27e5f2e25776d:0xc6366b2474a39c92', '0x89c25c5e135cce05:0xea520e466959511b', '0x89c24f89feea3265:0x7bceb7847f860cbc', '0x89d9f0c66929247f:0x20fbcfef9472b8c8', '0x89d0ac78660c3c31:0xb30bb912fa48008a', '0x89c2948a2c33f5f1:0x2d12acecd896b221', '0x89c25bade4f8471b:0x1dc68cc752a4c733', '0x89d1533d716f4cfd:0x7da4cfdb8001b01b', '0x89d9124a6e2de0f5:0xe218d4dc79faa74c', '0x89c25991e668c0eb:0xf9d157d0638b0407', '0x89e848c4f1b0e52f:0xc57b58169ea26678', '0x89d373ae5a5b89ef:0x10393a80f9f61f17', '0x89d12e353056226d:0x1f5b62a43f7bde71', '0x89c259f511996c43:0x5d4accbae88ee106', '0x89c25a10c1759889:0x497cbcf81ac0a2b7', '0x89c2585c35d9ec43:0xf37dd52a6cbc0135', '0x89c25b52a1fe1e3d:0xacd4e4ad0647d80a', '0x89ddbd0573b710a7:0x188b61946ec19489', '0x89dc2a004e8c93c3:0x9fdb602c904fa39b', '0x89db6baa16cb30ab:0x7f7727f884449fde', '0x89d36d33a0286f09:0x1cd3c162b79e10c', '0x89c2889c7fa7df3b:0x473c24a1cddc1761', '0x89daee54f00a6c49:0xa133714aabfe847f', '0x89dd4d77a96d5cc3:0x3e5b748bd452c23d', '0x89e83146a0b3e951:0xdbbd8684d1a59281', '0x89e833894e9f61ed:0x354807da7b4c3696', '0x89e836b1b74750f3:0x597ed9a817437c30', '0x89e82c36a468522f:0x69450bff795d70f', '0x89d1771679a1e555:0x8f51681d549494fd', '0x89c26ffc148b78a5:0xf52de62a593e2e6e', '0x89c25857e4c9aa8f:0xd17547cdcaa9dc17', '0x89c2f5fd03d38917:0x2681f12dade6cf8a', '0x89c2f60625e06fd7:0xebb242f312365c2d', '0x89c258a12bf93b1b:0xeca1467e81704b0f', '0x89c258829f17f13d:0xad008426e8a1cd37', '0x89de0d40f02f74f3:0xa941059bc35d8e2e', '0x89c260ee2d03a065:0x243bc99ac8398c7d']        
        google_review_newyork = df[df['gmap_id'].isin(lista_locales_newyork)]
        # Eliminar las columnas que no seran utilizadas para el analisis
        google_review_newyork = google_review_newyork.drop(columns=['pics','resp'])        
        # Elimina filas completas duplicadas
        google_review_newyork = google_review_newyork.drop_duplicates(keep='first')
        # Convertir la columna 'time' a formato legible
        google_review_newyork['time'] = pd.to_datetime(google_review_newyork['time'], unit='ms')
        # Crear columna con el año de la columna 'time' 
        google_review_newyork['year'] = google_review_newyork['time'].dt.year 
        # Extraer solo la fecha sin hora
        google_review_newyork['time'] = google_review_newyork['time'].dt.date
        ## Convertir los tipos de datos
        google_review_newyork['time'] = pd.to_datetime(google_review_newyork['time'])
        google_review_newyork['year'] = google_review_newyork['year'].astype('Int64')
        #Filtramos solo reviews desde el año 2015
        df_final = google_review_newyork.loc[google_review_newyork['year'] >= 2015]
        # Reordenar columnas
        df_final  = df_final[['user_id', 'name', 'time', 'year', 'rating', 'text', 'gmap_id']]
        # Rename columns
        df_final = df_final.rename(columns={'rating': 'stars'})

        #============================Analisis de sentimiento================================
        def limpiar_texto(texto):
            if isinstance(texto, str):
                texto = texto.lower()
                texto = re.sub(r'[^a-z0-9\s]', '', texto)
            return texto

        df_final['text'] = df_final['text'].apply(limpiar_texto)

        # Inicializar SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        
        # Función para realizar el análisis de sentimientos y asignar categorías
        def categorize_sentiment(review):
            # Verifica si la reseña está ausente
            if pd.isnull(review) or not isinstance(review, str):
                return 1  # Valor por defecto si la reseña está ausente
            else:
                # Realiza análisis de sentimientos con SentimentIntensityAnalyzer
                sentiment_score = sia.polarity_scores(review)['compound']
                
                # Asigna categoría según la escala proporcionada
                if sentiment_score <= 0:
                    return 0  # Negativo
                else:
                    return 1  # Positivo
                
        # Aplicar la función a la columna 'review' y crear una nueva columna 'sentiment_category'
        df_final['sentiment_analysis'] = df_final['text'].apply(categorize_sentiment)                
        # Convierte la columna 'sentiment_analysis' a tipo de datos object
        df_final['sentiment_analysis'] = df_final['sentiment_analysis'].astype('object')
        # Reemplaza 'SD' en la columna 'sentiment_analysis' donde 'text' sea None
        df_final.loc[df_final['text'].isnull(), 'sentiment_analysis'] = 'SD'
        df_final['sentiment_analysis'] = df_final['sentiment_analysis'].astype(str)        
        # Crea la nueva columna 'state' con el valor 'New York'
        df_final['state'] = 'New York'
        df_final['short_state'] = 'NY'


        #========================================================
        df_final.to_csv("tb_google_review.csv", index = False) # Nombre de la tabla que se creará en BigQuery

        return df_final

    except Exception as e:
        print(f"An error occurred: {e}")

def cargar_df(project, dataset, table, df):
    """
    Carga el df limpio en bigquery. Disparado por la funcion "captura_evento"
    Args:
        project_id (str): nombre del proyecto
        dataset (str): ubicacion del dataset de destino en bigquery
        table_name (str): nombre de la tabla de destino en bigquery
        data_limpia (DataFrame): dataframe limpio para cargar a bigquery
    """
    
    try:
        # convierte todo el dataset a str para almacenar
        # df = df.astype(str)
        df['user_id'] = df['user_id'].astype(str)


        # guarda el dataset en una ruta predefinida y si la tabla ya está creada la reemplaza
        df.to_gbq(destination_table = dataset + table, 
                    project_id = project,
                    table_schema = None,
                    if_exists = 'append',
                    progress_bar = False, 
                    auth_local_webserver = False, 
                    location = 'us')
            
    except Exception as e:
        print(f"An error occurred: {e}")

def captura_evento(event, context):
    """
    Triggered by a change to a Cloud Storage bucket.
    Args:
        event (dict): Event payload.
    """

    try:
        # Obteniendo ruta de archivo modificado y tipo de archivo
        file_bucket = event["bucket"]
        file_path = event['name']
        file_name = file_path.split('/')[-1].split('.')[-2]
        full_path = 'gs://' + file_bucket + '/' + file_path
        
        # Ejecuta el código si los archivos se cargan en la carpeta correcta del bukcet
        if '/' in file_path:
            main_folder = file_path.split('/')[0]

            # Especifica el conjunto de datos y la tabla donde va a almacenar en bigquery
            if main_folder == "google_review_newyork": # Nombre carpeta dentro del bucket 
                
                # Especificar
                project_id = 'sacred-result-412820' # ID del proyecto
                dataset = "dt_g_review."     # Nombre del dataset en big query
                table_name = "tb_google_review"  # Nombre de la tabla que se creara en BigQuery
                
                # crea el df segun el tipo de archivo
                data = leer_archivo(full_path)

                # llama la funcion para limpiar el df
                data_limpia = limpiar_df(data)
                
                # llama a la funcion para cargar el df
                cargar_df(project_id, dataset, table_name, data_limpia)

    except Exception as e:
        print(f"An error occurred: {e}")