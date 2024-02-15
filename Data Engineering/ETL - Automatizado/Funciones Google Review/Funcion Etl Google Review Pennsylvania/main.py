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
                df = pd.read_json(f_path, lines = True, dtype={'user_id': str})
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
        lista_locales_pensilvania =['0x89c6af439d9163e9:0xb4349e5d37d7ee75', '0x89c646f3f822e73d:0x19bfe62785468634', '0x89cb0fe2de411611:0x2b4f53e4b86f9dcc', '0x89c95f209af7ab59:0xdfd166b6d98022be', '0x883492835f080b59:0xd3db66c6fcb7f587', '0x89c8c104954c6009:0x29a7555176df9878', '0x89c6b8ef07039a49:0xe406d7ea09b6b6e3', '0x883473c1e626c419:0x8a1d03cfcaa642ba', '0x89c6246352586b1b:0x2ed559577eb35972', '0x89c5ec6267ae4963:0x3a2d9f2ef33ca94b', '0x89c69374c1d3bfbb:0x8246523a9a0a4f36', '0x88348b5034850fc7:0x880fafc221135544', '0x89c6730e606f37e9:0xb0007fa3eb631d2e', '0x8834c30dcf9caac3:0x62e921a2d932d97c', '0x89c69370f408af49:0xecb2a289c404fc8b', '0x89c6b35b637f801d:0x9c9f0d69e71af8e1', '0x89c430833cec866b:0xa6f8232e75f98a2d', '0x89c91130a86aaab3:0xc249c58f5c7543e4', '0x89c152a5c0000001:0x5bea19b6fd97a54b', '0x8834738e6d775297:0x2e9359ee2e31d968', '0x89c99efe108eda2b:0x82a57993dd373391', '0x89c43175453e66cf:0x8fb1a3c240d87fbd', '0x89c8894d9599c56f:0x3c986e60300d9af', '0x8834f6d38c511a6f:0x75179251835267b6', '0x89c8ba52a37efd2b:0x44d7f08a32e6a989', '0x89c8e80841a7670b:0xd9e83547cb35d064', '0x89cea8a231ce7df5:0xbc75ecf38a12d70c', '0x8834f156cc60c6f5:0x88d9a1997baec40a', '0x89c6e7e2f0f9c29b:0x53bce5d5f48006ad', '0x89cb5ad7e20d0433:0x13d7fc042beeb042', '0x89ceb66ee39bda1f:0xeb01a303cfb46f1e', '0x89c4487e07028f6d:0xb42f232235cad408', '0x89cb5ad729e1f659:0x9ac38f3a361c666c', '0x883448eafa3eb433:0x8e6d929ee45b764f', '0x89c614351c2aa745:0x1a2eb612641086f', '0x8835158bf8de2c23:0xcdc7786b3e19c720', '0x89c6bc47bfa570e5:0x132a429b0e2abde5', '0x89c68e0aac7ed797:0xe92e9eed769bce67', '0x89cade8c16f0d067:0x3125bf86b3d93051', '0x8834f15b91a18719:0x15d9bb6a7976ccb5', '0x89c5a71c4e21a479:0x98ff91b4fb65e151', '0x89c4479407d6c10d:0x5e5a2564a08148f9', '0x883391fafa61390f:0x27616fc41fabb790', '0x89c6b7802679b673:0xcae94dbbd1c0b1c4', '0x89ccfa514abb3e0d:0xd4cd61f854fdddd0', '0x89cb10b02e884615:0x63fb7ae9af1b3c76', '0x89c6224c97d8308b:0x6a2e5e17fbac6d55', '0x88329137ed97c709:0xec3df775c929db1c', '0x89c6c63144a64dcd:0xbdc95a1c008096a5', '0x8834fa0414bb22b7:0x62c949f9e43cbdd6', '0x89c60760bc5bbaeb:0x2f2f15726ec3e268', '0x89c4226a4af60e7f:0xd715b9e5b44bbe92', '0x8834e8d6762415c9:0x722f210431673d31', '0x89d00948276d652f:0xf125bfb3b6bacf3f', '0x89c156e518312b49:0x162c51323e72247', '0x89c88fa6ee12c905:0xfc5cce1c7ab5a664', '0x89c5e9bf430e6735:0xfab36c81279dd0a9', '0x89c696b2df706743:0xe3caf8ccb9212572', '0x89c88f3a4bbf4237:0xdf4ca52107f37d', '0x89c8653646b1878f:0xa09e56c292d34863', '0x89cb5ad165178477:0x34ffa76b474fb81f', '0x89cb0ab0fbb82bf3:0x8aa28cfa3ef1d187', '0x8834ebd149aeb969:0x4bbaa24b33d8e090', '0x89c6c7d3edf40203:0xb4e8d22f15ad18a2', '0x89cfa20212736c89:0xd6438517aba14062', '0x88346ff2d1dee04b:0xe98fd7d28da43e91', '0x8834d2cb82196a99:0x5fc0a2ed82510c5f', '0x89cac024097b5e81:0xcd08b7c8d5f88a53', '0x89c63b5c74b2c1df:0xfe6c29f2b5a1c317', '0x89c6c49c978d0d3f:0x1527af5910362426', '0x89c4b374e3d589d7:0x1d5f78f42a1ed669', '0x8834ece7a48f3bc1:0xa3ad0c3dd96413ea', '0x882d7e28bb7d7f71:0x21376f6b8474eb90', '0x89c685df2aa41c9b:0x7f69ed363333d382', '0x89c8bf3d6c91d273:0xd906de92f6cba610', '0x89cb21ce4e5184c5:0xede2142ce908f0d2', '0x89c6b3970f386de9:0x911b4b7ffe1c4f36', '0x89cb91396a82df11:0x81e81160159230f7', '0x89cf9eeaf462339d:0xd4dc8cb88eb30bc8', '0x89c5bbf6e002790f:0x9ca75202acda96ea', '0x89c6144a5f9e2eed:0x499dddf663bc865f', '0x89cb917deee97f95:0xd5e33a8be7798d06', '0x89c58381e0f14145:0x52568156eb9e05f6', '0x89cea5b4a71613a9:0xe71ee74d11ce2a7', '0x89ce19f9f887be75:0xde5b668e4660eadb', '0x883261465487d8e3:0x2c244a3aeb56be43', '0x89c4396b3466aef9:0xc56bfcb0b4b26d15', '0x89c4476d4f2d78e3:0x68f2010311377218', '0x8833333530933f57:0xf4dce1707c849bbe', '0x8834e219606bd991:0x46e64e931c009308', '0x8834f2474c314d51:0x6a3fe08327508e9b', '0x8834ecec6389b9a7:0x8a3aa7f3d07d0ff8', '0x89ca525992724a03:0xb49a0c8c3c42577c', '0x89c67bb1bc40c551:0xae58e6f16e985dee', '0x8834f17676b3ebf1:0x3333873044f88c8f', '0x89c4d8fd15428c1d:0x225daa5ad6cbddac', '0x89c8c20e5e2d89d1:0x964ebf683604d240', '0x8834595974a74ee9:0x62753750f5efa43c', '0x883324faaaaaaa9f:0xfc66643b6491c4cd', '0x89c603acadd0156f:0x55cb82dcdf8b66ae', '0x88351585cea3efdd:0x638e6e0b9b91eae7', '0x8834ebe1a5cfa1eb:0xa2061cc06a35ede6', '0x89cf6afcbb81109f:0xeb25785063f38b0b', '0x88351d0f7d246937:0x7d5806432a5b3cce', '0x89c5ec1c888a84fd:0x640ab2fc7a13a90', '0x88337482b279bab3:0x668000e1b6d23c34', '0x89c6029f151742d9:0x4a05dbd311caa0e5', '0x89cf5e35d6d746f3:0x11be89a428d24585', '0x8834f5cb706ec5cb:0x29e6ec212fd715c5', '0x89c4309bafa20211:0xa6e3564ca0179c80', '0x8834f1567b85b9b3:0x48c5999f620b0c4c', '0x89cb173238e495af:0xc07c0075546cae6', '0x89c3450aa77eb779:0x387ee2ac15595548', '0x89c696fc49c0cac5:0xdc1f10326ea431e1', '0x8834c5f01c023ceb:0x1689bb9cc8c76f8f', '0x89c63b69febc59d9:0x55c97e33cbee1997', '0x89c6c6edbd5768fd:0x9e1598c5cedb059e', '0x89c602e8601488f9:0x9a00409b84125feb', '0x88351a9f013b3275:0x5780bfaa9b726c0c', '0x89c90c85a15ebf55:0x5627d97c8c567558', '0x89c9baaa9e1b802d:0xb4961c1824504593', '0x89c5b36a8ce13b5f:0x4f5fa14a166914ac', '0x89cf514bbb3145bf:0xb62c752705fc919', '0x8834f65fa3749405:0xefdee3fb79c69d99', '0x89cefa13fdedcd85:0x3b23a4f65286568', '0x89c6818faef7c527:0xc0a882a342700772', '0x89c6e9d9931ab3c7:0x74076c38ca5dc47', '0x89c61698e913475d:0x7744e2be9480c534', '0x89c6e801ec54ee87:0xd71eb257e3b9d9bc', '0x8834ff4590aa3e75:0x47515e35596c304f', '0x883280a4fd804907:0x66189818eeecd7b1', '0x89c8c19897f16b7b:0xc9404b6811f00096', '0x88338dcfcba651e9:0x82bde1797dd156b7', '0x883466fedf342171:0x884f903e806bbb16', '0x88338db0c86c4671:0x7ca0ea055585d0cc', '0x89c60c8e0e5ff641:0x28fed5b45b023494', '0x8834668ef336d721:0x139fd075a3cad5b8', '0x88345dca9942cdef:0xcd842f4f3e1c9f90', '0x89cf700c3601a7f9:0x6315e2f05d657387', '0x88335d62b296718f:0x37604504fc22c8dd', '0x89c43f2c5fe4fd15:0xa88022d29dca89cd', '0x89c8509485e56db7:0x4854aaf570c9cab4', '0x89cbbd78dd7aa50b:0xa22d6ce26746fe0b', '0x89c6f4d91d0cdfc9:0x5ef43640e3aedef9', '0x89c6e9f12ea3bc99:0x7845957883c0b55e', '0x89c88c3961bb89ed:0xc756f2cb0c06602a', '0x89c4699b75241249:0x5a42a7188cf14e24', '0x8834f24d100685bb:0x190fc559f866f890', '0x8834c5db9dd99589:0x82d569de234815ee', '0x89c894c593a76497:0xd158d8af2185b2b8', '0x89c624ed94235303:0x11cb6179c3170015', '0x89c8ede3a912d9ad:0x2f9467661f6e72f4', '0x89c898fcc9095443:0x3e46eaa9aaf1f8b8', '0x8835a02b796d4827:0x97d0ad6f218ddea9', '0x882d67c8bbca9bc7:0xdeb4e68c56f38db1', '0x89c6c6fbb78bb28b:0x45d6263e2833570a', '0x89c44cd65fbdc4df:0x5769087f2d4736a3', '0x883281def0e46eb1:0xd43738602c0b9686', '0x89c5d8f3b7b2ae15:0xcfc0b825ba2cef22', '0x89cb101375015197:0xecbed8a44c21ec18', '0x89ca5255f92f7657:0xd09639aa900593db', '0x89c88f0edb782699:0x2abfd83c9878e557', '0x89c4dec267d46a4f:0x53bbd9f71e496b8e', '0x883250e1dd2f8665:0x4078b19ab2224455', '0x8834961d53392ac7:0x9a238c9c8487f1f7', '0x89ca52d0480400f1:0x5d8cef76e9b5b1ff', '0x8834934c53c6eb09:0xd43908124a44b680', '0x89cc66f6e369c87d:0x9b7a07cbfe94eafe', '0x89c677a3bf98acc1:0xff97ad27c998da92', '0x89cbbf16d8841933:0x6ad1afa301994e4', '0x89c6b06188944557:0x9c313809b6bf53', '0x89cb73f5e7a8deb9:0x2e517cfde2339295', '0x89c6babd3c8e0411:0x9c5e7fb201fbd741', '0x89cf7db05ceb7f5b:0x9e1a9a729a08e2fd', '0x89c6ae9730efb75b:0x9bc7f0684a3ea281', '0x88345e1289be5c27:0x5560cbffa13223d0', '0x89c6af4ea7319e99:0xaca16121c77685d7', '0x89c433dac60ece93:0x72a2f4058b8a1f90', '0x883467c107ab59bd:0x9b79a2110f56cd16', '0x89c4884933f2e99b:0x1785ade624701da1', '0x89c4a6bb6719bff9:0xecdcc1ae91908d24', '0x89c5e0caecf39fd9:0x91112f14fe78afcb', '0x89c6b708a7bc04b3:0x395d6848fba3a1eb', '0x89c5dc99bc793acd:0xc47e9ea5d21b7051', '0x89c624a345497c11:0xae9c5e7425a9066e', '0x89c8c40641bb054d:0x18c717cd11cb21e', '0x89c6b0821c966dcd:0xf93b93ef62e3456f', '0x89c51bad6e793931:0x587302ed759b1fab', '0x89cb96e5416641af:0xbb1d541ae355de70', '0x89c6132d1b0f31f3:0x2b96f4491956b3d', '0x89c5877b796aaaab:0x96ccdff983fe162f', '0x88345063d64fdc97:0x52774dde66cec150', '0x89c51b8913ff3321:0x8fcda8c2be25c828', '0x8834f9913c5b5489:0x874b2d87edaee8f0', '0x89c4d6925a155ad3:0x5592472ab713aa57', '0x89c90c85a15ebf55:0xd067b4bdafd839e', '0x89c681f3590a6ca1:0xfaa199dc4f9f208', '0x88346f2b0d930ab5:0x1aeae3b27a391244', '0x89cb675211f139e5:0x9294b5cfe7bac3cd', '0x8834872d9c3d3eeb:0xd4050a582382456', '0x88349596869a99fb:0xf02e3a87bb022c65', '0x88347d2123152545:0xc3c79b6d83ddd80f', '0x88353e8de85c4ed1:0x9eefa16ba1752b77', '0x89c670dd3ee9f279:0x317680279510cf97', '0x89cd0777d10cf0c7:0x60c3d66088b8d9a9', '0x883376f810f1c00d:0xcc2c1c5b524ad969', '0x883457fa82d2d3ad:0x733e5341b5e2dbef', '0x883396ce04ad8829:0x4d4799efc6383130', '0x89c63b4aa001c581:0x240362d4eae970ea', '0x89c14d3b7f1f0d31:0xc40509a6f1f25203', '0x89cd4ec3aa036aab:0x709cff3ea5205ea1', '0x88345937bb7befe1:0x1475b53024a4f583', '0x89c88908a01c5a73:0xe8261dad347bd443', '0x89c4407d8b77808b:0x900c932cd4a45a5c', '0x89c6c62eece30e07:0x9996f86a2b7c33de', '0x89cbf585e34406a1:0x9ad0c7f7d887f161', '0x89c6238ae148a051:0x8d86f980e28075c1', '0x89c622f03d30d00d:0xa4f6bcc625d25b61', '0x8834eeb0d48766a7:0x3c6be46782d74035', '0x89cd0d0a7d2f8ea3:0x1333dd9c893b4005', '0x89c41fb2aa8ce223:0x12ec5641d1080aa0', '0x89c9904fc7000327:0xa8fc48855ab57f91', '0x88348fca58bcdf43:0xf531dcbf8531ec', '0x8834ea3e84d04cef:0x3227ffca3a6d34b3', '0x883455ae4b8c5f57:0x57f833719ac3702', '0x89c8e1d9247d7bbf:0x45c0bd3e7dd4e739', '0x89cb1045e3e534d9:0xc68bb00c99773212', '0x89c4d95e455a13e9:0x67d8673802405c90', '0x89c4ded692ef7da9:0x18dc60b0ce215471', '0x89c4d92277e4e043:0xa18d57b228a28edb', '0x89ce2eec750001bb:0x4b9589e84f8142c5', '0x89cfd63d900844cb:0x18d9df417f40b5d5', '0x89c84e0682a1d74b:0xe6ee8b425686f65', '0x8834da9f9595c50d:0x677b397948c10ab9', '0x89c99e9fe96bc385:0x86070272f572d1cc', '0x89d00be11a8016eb:0xc52408269e0547d8', '0x89d0094959165b11:0xcbfc7e9b2ed30908', '0x89c43c5d9904df57:0x4352b13f8a501eda', '0x89cb8e1de35c94ab:0xcd2fcef11857da05', '0x89c57473ddb9e91b:0x9fc5976bd9f5fac7', '0x89cea7d5a327bfa3:0xabc28bc3f21e0b7e', '0x89c886518c610f1f:0xa87d63c6376ca6db', '0x883383b3ea5f18f9:0xcf20c53ff7d508b0', '0x88351ae3ea1c7c29:0xf2f6d35e86ca2e06', '0x88327e270070de11:0xa812bd578515af3c', '0x8834f84a6280c12f:0xd8256b1d033d7fed', '0x89c8e0412ca54b49:0xfd785d4c1cb8755d', '0x89cef13f536f7359:0x2d83a5e2b745509', '0x89cbab919afc6f5b:0xb0500a7ddbd87414', '0x89c488c09e85cc89:0x306d1dee28566027', '0x89c6c8ade5f1d0a3:0xa52d77b24b05aec6', '0x89c8a89d38d20b4b:0x989b771fe04010a7', '0x883307fbfda2c145:0xc833bcee7517b441', '0x89c6be818af71d37:0xb0535c19fc1e75e4', '0x89c85a1c488d03bd:0x992f99846347bc92', '0x89c9bdbf23cdcc37:0x20c34f1047853b1d', '0x89c9981b5a8f7219:0x597305d4c7d7b457', '0x88348d2a39ef30a3:0x1a6b1af9b721d1d1', '0x89c6f48883e3f34f:0x262a978c955f05a3', '0x89c6f49c1451704d:0xa3b66c5fc4c6a5fe', '0x89c64cbe62a3ad49:0xb749d9f827681212', '0x88351ce9e2cfe3fb:0x7400ce9ffc542a19', '0x88351e21ecd1e3e1:0x7efef6c2734ca9fc', '0x89c6c64edfc6112b:0xf10180310a37b2ea', '0x89d00c29bb13d725:0x15832be4370cab03', '0x89c4cdeb396305e5:0xd2baa39f53657c34', '0x89c6c881f35f09e7:0x846a2b70a97ebc07', '0x89cebbc13dd66e41:0x12a4a9d9f86a626d', '0x89dada7d6705ddd1:0xe53e232f91561270', '0x89c6a39209724b19:0xf187f9423328c5fa', '0x89cd4cb233aeb793:0xf85a7e9922695d77', '0x8834e92f456734e5:0x2b7f0f2d2027f3cf', '0x89c8cfdfe37116f9:0x3f4b66c170e0409b', '0x89ceb550cd046d7f:0xf569d0511c96b425', '0x89c8c116d1153b25:0x2f78742801a200d5', '0x89c61f6f70d26d2d:0x33e2ea8940eb79a4', '0x8832c338adea9e31:0x6e082a7c99aff2ab', '0x89caddad2e618d5d:0x27dd880e5bf72de', '0x88327fdc6147f2f5:0x80842a92e7f5eda2', '0x89c8e9ceb1bf0547:0xb5785176b3378f9d', '0x88346ff921b60511:0x79f42a55d47d20', '0x89ccb3bcd5a0e73d:0x928a48b660672f36', '0x88333fd64958172f:0xb76cba84f6ae1502', '0x89c592da20bcfac9:0xea85c670b33ac90', '0x89c431e9d069480f:0x21ba2a99ddda23c5', '0x89c4f2f3bbb8ba39:0x7e83ce1f5baa945b', '0x89cf6d06110f8991:0xc1f1cfd39323a9e6', '0x89c6855bf0895b37:0x33ea98ad9e6f02ae', '0x89c8cea90706aea1:0x10b5b8da678c31ad', '0x89cb52318b4b82cd:0x574f0d736f5c8484', '0x89cb5b26becb1015:0xe2d4e1e510f99544', '0x89cfd65750905ca7:0x8236c89e2936950c', '0x89c154ebba3eda05:0x5613bb777dd7cece', '0x89c46c3c5de4ed15:0x16a76415580a7881', '0x89c5a6b1c8abf88b:0x6fcc7a57685db88f', '0x89c6c6090f3ff59f:0x6e2593e56014c46f', '0x89c644e485f40347:0xa57c206c58c6c74f', '0x89c500f02ad94353:0xc915db0a873c2f23', '0x89c8bf74148c26a5:0xc26824ecbe7de743', '0x89c89e5f27fa8b75:0x16a334e57da33a1d', '0x89c439ade5305313:0x3d07cde1143e9bbe', '0x89cea7da357ce9c9:0xf8a628e7ee3a410c', '0x89c485b4bffc5d5b:0x43112d32f068e03c', '0x89c6b02fded0d88f:0x352eb1d0f66ea5e6', '0x89c5d3c41c99871f:0x454bc8fd293c8627', '0x8834dd242cb63e39:0xb81a371b7e22766c', '0x89c6ae8c65c2521d:0x519468cf587e6602', '0x89c8e826c69cfb57:0x40bb5ae3dc67025', '0x89c8c28af6675ed5:0x1f2b76ad65c1bca', '0x89c673d081738acd:0x2dc63b7047eb08a3', '0x89c8bb3a56c1577f:0xa98d106e3471382a', '0x89c9ac8e676d9317:0x6d4cf5d7d4cee4c5', '0x89cfd1c9ae5d998b:0xab15be077481f706', '0x89cf19c10c14d447:0x1a0a585d8f0f5a64', '0x89c88bfb72bccebf:0xab32e47c496a4a07', '0x89c94b033f9d9c83:0xd8d60a34f501e6c5', '0x88327034810c8d2f:0x52e93763eaaf15ed', '0x89c69693a980e7bf:0x6e45ca3482072910', '0x89ceb5622092c1d7:0x8f7f7fc31ff6ffed', '0x89c4d993f2307b5b:0x7d0afe7a43de2bb9', '0x89c5058531549fad:0x6874fe4036edafbc', '0x8834d087c2500197:0x45253a3bbd194d1d', '0x89c67d0e236aee21:0x8377babf25538725', '0x89c602c5f25cf0d9:0xf13c334a54784fe1', '0x89c8a86dbf52d4ab:0xce3c33ff15c9e6d9', '0x89c43e522db55303:0x64976b0ce3af5f33', '0x8834e8f60d014843:0x2a9d415a2ba08138', '0x89c47cbe886ce23d:0x7973b41acc2a16bd', '0x89cfa63be81ab691:0x4966075e6883b169', '0x89cf4aff57994a95:0x7d5b3ef0920021f2', '0x89cbb72a30565131:0xa21308732803b7fe', '0x89c6f8f4670b46a9:0x2c5db6d0394e0ad2', '0x89c894d136223bfb:0x26d26431756d066e', '0x8834d9ed138634bb:0xbc959d955c5ac0f5', '0x89c4d0f6fadaf47f:0xbf7895eea0f38847', '0x89c4d9965a77cfa9:0x5f3392c6c1c9cc5c', '0x89c51a8a70a76273:0x4c75cc60b6ade128', '0x88350cd65da32221:0x4f8f905592670242', '0x88346d9bdc0d3537:0xefd08d91c8b1ee16', '0x89c34599ef7b54c7:0xdce26b959cf15ae7', '0x8835a1e7357103fb:0xec2f44db18d7d73', '0x89cf0d694e0fe9bd:0x716e46f709fb8f4', '0x89c4fbdf27637dad:0x5e08eb2c0ae8544b', '0x89cca9f11faa095d:0x711625424c920ab5', '0x89c51a84800bcc73:0xc5f6f4e06530b55e', '0x8834dcdefec44d59:0x24f71fc86974401', '0x8833c070324c6589:0xe52bf3c56134973b', '0x89c51a30ef7e5f1b:0x2b5c6f71ab1dc3d9', '0x883477fa13440927:0xb8a326f70cd53bd1', '0x89c437fb1cd612eb:0x2039b64f78865329', '0x89ca4d7248d4406d:0x59b35fbff31e97ba', '0x883383c58ee1620b:0xd382662f817cef46', '0x89cdac0951890b57:0xcb618750af4d9833', '0x883494fa1521bf0b:0x92c22990dc749c97', '0x89c895f556164d3f:0xf50f23e6cb6287e3', '0x89c6afb0a2a57173:0x6ab24f0da8c056ba', '0x89c4161f9185b27d:0xf33259cfe928f639', '0x89c88ef95c5e6da3:0xe014fc0bbc415460', '0x8834f23a8d6a64e1:0x1c0535b9834de21f', '0x89c63ba6e8f960e1:0x2f775e4f81983cf7', '0x89c58d2a68e01d33:0xde52ba1f33356c81', '0x89c6c623e20fc3bb:0xd266e21689cfca00', '0x8833b13417cabb7f:0xb0b504f1882ae88a', '0x8834f15e5c235fab:0x98834108e3c4335d', '0x8834ee2f4fbdf101:0x92ea5e37b1754a7a', '0x8834d6f3cb6bf717:0xe9f827e498ea6dcd', '0x89c89ce2ebfe6e77:0x8ff8393beff17ecd', '0x89c6f24e9358ac5b:0xf348fd7a2f8a03d', '0x89c6712a1420857b:0x706d41ece8daabaf', '0x89cb16e98268b7b1:0x44364f3dd485282d', '0x89c9ac43007e23e5:0x8b3fc8a75dcbfdae', '0x89c6c62dc184beed:0xa6f2d5dc52759962', '0x88345fb68fff11ff:0x662b15ec56a6cc60', '0x89c4410941fca221:0x67492ef04be9cd5e', '0x89c88def39dfb2ad:0x59b71251f5d30637', '0x89cfa67c70833487:0xeab7f785e1193d0d', '0x8834e5d9706d1021:0x40118759b8b6efc1', '0x88327e555bfd05e7:0xb9b7a9942b4ac4b8', '0x882d7c571c7fb2a1:0x9b1fa90739703e5', '0x883457fa06f496c7:0xe1957ed94278b5ff', '0x89daaa86c73c7db1:0xb428b9f71f7eb64d', '0x8834f77631628b31:0x2b73ae1c5cbf256a', '0x8834994cb5317fc7:0x1d3c30a53edcf6bd', '0x89c6944bbcf4f9fb:0xb90f9769e59f62b1', '0x8834f9a714946727:0xd247b21e468f06f', '0x89c676a409213c21:0xc4a2c8cb4b2a5d8b', '0x883261914c3daf3f:0x94130ce5345c8cdb']
        df['user_id'] = df['user_id'].astype(str)
        google_review_pensilvania = df[df['gmap_id'].isin(lista_locales_pensilvania)]
        # Eliminar las columnas que no seran utilizadas para el analisis
        google_review_pensilvania = google_review_pensilvania.drop(columns=['pics','resp'])        
        # Elimina filas completas duplicadas
        google_review_pensilvania = google_review_pensilvania.drop_duplicates(keep='first')
        # Convertir la columna 'time' a formato legible
        google_review_pensilvania['time'] = pd.to_datetime(google_review_pensilvania['time'], unit='ms')
        # Crear columna con el año de la columna 'time' 
        google_review_pensilvania['year'] = google_review_pensilvania['time'].dt.year 
        # Extraer solo la fecha sin hora
        google_review_pensilvania['time'] = google_review_pensilvania['time'].dt.date
        ## Convertir los tipos de datos
        google_review_pensilvania['time'] = pd.to_datetime(google_review_pensilvania['time'])
        google_review_pensilvania['year'] = google_review_pensilvania['year'].astype('Int64')
        #Filtramos solo reviews desde el año 2015
        df_final = google_review_pensilvania.loc[google_review_pensilvania['year'] >= 2015]
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
        # Crea la nueva columna 'state' con el valor 'Pennsylvania'
        df_final['state'] = 'Pennsylvania'
        df_final['short_state'] = 'PA'



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
            if main_folder == "google_review_pennsylvania": # Nombre carpeta dentro del bucket 
                
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