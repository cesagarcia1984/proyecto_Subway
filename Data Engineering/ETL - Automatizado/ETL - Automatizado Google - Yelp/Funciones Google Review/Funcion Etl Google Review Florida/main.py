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
        lista_locales_florida = ['0x88e76652cd84272f:0x548abb9935d912ff', '0x88e70d7d46700a23:0x8a9191a6c55c84fc', '0x88dc1f86a88dc809:0x13c0c1bd2c1418c2', 
                         '0x88c2d2779608fa4b:0xbd9be2942baa3b9a', '0x88d9b6811a7761f5:0xf3d18ce005b495a0', '0x88dae1e9c04de3e7:0x7543cac1b9b0829b', 
                         '0x88d9012935620241:0xb3a9b5b0671e51a5', '0x88e7240eabdff42f:0xa59118725edd85e0', '0x88e5c479257cf7e5:0x202593b75d7213f0', 
                         '0x88e7a06be791aa09:0x67394c10f5deb3ae', '0x88db1763826c4743:0x277418ee319b9a37', '0x88ded664dbc87107:0xaedb4243e35923c2', 
                         '0x88db6f20a6f4a265:0x2566dc0377e83ec3', '0x88dcabc879fde7cf:0xd96cb81cabaefe90', '0x88d7661b67d14f4d:0x2cd1a23a04165eae', 
                         '0x88dcaba6847cb39f:0xaea6e55f3c742dcc', '0x88d9a5ea448dd88d:0xe04def60536fce85', '0x88c2b7181c4deead:0x29da2101a8ab1e42', 
                         '0x88e5c96bf67d165b:0x63421607eaf553c6', '0x88d9098be58f3c81:0x8a0217c20b6b0d6b', '0x88d9a10acee84987:0x66f971dba6bbf038', 
                         '0x88915b8cc7d523ef:0x3c20a76e860ea335', '0x88c2f29c89fa1759:0x2c449ece7716834b', '0x88d9b99dad3337b9:0x9c1b2358bdf46fc', 
                         '0x8893d1bf97082aab:0x95428447be68e8d9', '0x88dd395e511ffbdf:0x5524997d70ac3a9f', '0x88e7828affc3dd83:0x3feb96529ec5afc3', 
                         '0x88dd6b1ea2efa053:0x862a629384ffb16d', '0x88c2d01fb50d9deb:0xa95526f69e935685', '0x88d92a864981f961:0x118cb474ee6cb6b0', 
                         '0x88d9ac56ad1798c9:0xc4831e8b7a97063f', '0x88db140d80b12157:0x954e0ca9448acab2', '0x88d901ff9429e451:0x6b87e77599a3c103', 
                         '0x88deeddf54b96dd3:0x8a2f48efdb1ee5e1', '0x88db41e406b27299:0x597848605bc79371', '0x88d9a86254ad6111:0xa697b33914fc4a5e', 
                         '0x88dd81bb9378cc4d:0x38c236d0f831b22d', '0x88e5ae504985394d:0xa5eaecea3fd83e8b', '0x88d9ac51c5140e53:0xc2dad6fa269faa54', 
                         '0x88d9a65c8c72d8bb:0x6920dc792f0d52d1', '0x88c2c735534cd613:0x70c4f2e073a42d7d', '0x88e5bba1444dcfb3:0x40758796619123d4', 
                         '0x88e7cd3fbb0d23b1:0x8e11b225c4847ee7', '0x88e8a39d05d55ecb:0xfa9a4370ca5a9575', '0x88d9bc74bd3d15b9:0x238ad8199b7638cf', 
                         '0x88c2c3a723690f5d:0xa880da443fa72055', '0x88e5ab309ea5c375:0x51665bdf2a1d2f5a', '0x88ecf2ee04aba347:0xa602cbbd54759ced', 
                         '0x88dea01f57fee89b:0xb4b8ae413fbac', '0x8890b9578f99af4d:0xd88ba7a29abcd9a6', '0x88d91d1ae46930ed:0x8532b234f747ade1', 
                         '0x88d9c0c3a05a45df:0xb3d1f93ccfb2da60', '0x88e895aa09c60e47:0xc70f17ecb5278841', '0x88d9b9ddefa3b34f:0xc6503d556b36a2fa', 
                         '0x88d8d649957ed9b9:0xd32a69206726db27', '0x88dec4cc8ce244af:0x5031e85ad5e42075', '0x88e7c625cb4a82bb:0x94400006a6bd830a', 
                         '0x88d91c3a65a4d7bd:0x74275e15263dfd1b', '0x88d909b7198972f1:0x9e2faeacfc45fb5d', '0x88e71be5aa67fff9:0xe688865c4cbb6742', 
                         '0x88d91b9091ec974d:0xbaa2cfde76f66199', '0x88dae24cf7c8dbb9:0x1982a0980115b021', '0x88d8fd8982048c59:0xe7abec5249ea19a6', 
                         '0x88ecf547cf456365:0x4bbe5ac5ed9e9730', '0x88e7819c2ceace03:0x6610e2832a9662f9', '0x88ecf5aa6260a8a9:0x6371b62847d221b8', 
                         '0x88d9bf1ec0800db9:0x80708a99a8c6ee72', '0x88e5ae52d5897dc5:0x80a1683a82516f83', '0x88dd874ea75ea013:0xec58d1bcfdfc9f51', 
                         '0x88dbfa4e7321d1ab:0xf92ca09a131ec1dd', '0x88ec5eef7ead7877:0xe23d561a5de47390', '0x88e0a9d63a7add11:0xa012b39d0d85c2f9', 
                         '0x88d908fc3fb2bfa3:0xc70991f26b1d9c65', '0x88df2a97e5e9b855:0x33e74f0717afd303', '0x88e7c355bc4c6607:0xd06cd8a62732a180', 
                         '0x88e5b0b49bda8b3f:0x7aa6bf88b25b2118', '0x88c4aac8ecfc98dd:0xeb22dbcaa86ac8f1', '0x88db1c3a5e46ce61:0x6856854e63ee01b5', 
                         '0x88dd0c4ab8e7293b:0xbdc36501cfe1a05b', '0x8890eab3fddbb743:0xbacb60df3573f41d', '0x88d9a67a4762e5fd:0x9813c9676fd76617', 
                         '0x88d9a6c8e25c472f:0x532ffb27f5cd0a7f', '0x88c2fd3527fa56dd:0xbb98d8b492f1359d', '0x88d9b796a7263fab:0x427e9ab8a82367d8', 
                         '0x88e7408c82a4bb65:0x1c5ea138a9cad67c', '0x88d9a8d9d4d7c4eb:0xca206694f94dbf8f', '0x88c2e28e9522de47:0x5fe5164ab168a0e9', 
                         '0x88eede9cb0f9b03b:0xc700fa8d0d97da27', '0x88da5b7f1289777d:0xd47c3d1f6d778d71', '0x88d9b2dd5feca155:0x4b3cc5e5e1e40346', 
                         '0x88d92f5eb47e56f3:0x2904c22345fea63', '0x88ec7ce85600f4c7:0xe0e92be2cb12ec70', '0x88d920220bd661f7:0x5342dea414d50e66', 
                         '0x88d9508c1e83bb41:0x300c8ba61cec845f', '0x88e717949ae55937:0x851db60e9db92d74', '0x88e77b0f3cc5ebc3:0x86772ce582795390', 
                         '0x88e62c13dd3d625f:0x7c8250284aae4582', '0x88d9c7a4e23da1e5:0xbedd23660b1e0b08', '0x88e5f573dab655fd:0xf287ec407f097578', 
                         '0x88d9c30852ccb619:0x198931153c38f944', '0x88d9b83e0fbc5657:0xa86a9b2eea1517bf', '0x88c2f017ef818001:0x1c0c47acbba308dd', 
                         '0x88e5ce8aa66683f9:0xf271b13b54af8d21', '0x88d928843231a9d1:0xd6909e6c31e44199', '0x88e778c48e4a8891:0xf226d9635549319d', 
                         '0x88db59b14136ebd9:0x7e1a028b79924d16', '0x88dae29eb24fd9d1:0x82ad97b6b90947d1', '0x88c2f98215d9e4e5:0xe055d49f6c7a855', 
                         '0x88ee90bb3048baa3:0x45367a81e5a85dc4', '0x88d9c36ebb30c37d:0x97527895f7c6cec9', '0x88d9b15d0e3ce6dd:0x8d86dc77a8b2eff', 
                         '0x88db68f09c5d5177:0x64353a20584ebd83', '0x88d905b4509856a5:0x8c6591ed80fe58e8', '0x88c2d6fa6527956f:0xfbbc109fd1c00c17', 
                         '0x88e5b86e8cea0e8b:0xf4613bea3a0d2108', '0x88d90066f0d272dd:0x771d5dbb68c8a35a', '0x88d8dff96a42d073:0xd0ff31dfd643a0e6', 
                         '0x88949afe9d1572d9:0xbc4033c5166b916b', '0x88c2f2a5ed7904e7:0x82d3e71132f3710d', '0x88d9c7a6a9236f35:0x3513b4b0502fecc4', 
                         '0x88938ed2676660c3:0xd6a550d68b6a88d7', '0x88db03d65d36b5df:0x624155aaec5c4800', '0x88c2ea3e01c05ccf:0x625a4a40de479d16', 
                         '0x88e684cdc5b5c00f:0x9b04bf96bef01a2c', '0x88e725dd4c2035c3:0x78ccac04b00ac311', '0x889141c5f2b1ad91:0x215c52c3437b3902', 
                         '0x88db439a812acadb:0x4075c04c666d0e', '0x889399315ddb4501:0x3a9d4589638cdeeb', '0x88ec85f9670da237:0xe1446e830078d456', 
                         '0x88d9a9952843331b:0x675ffbf217063226', '0x88e79d31e7434ee5:0x42deb1529a024def', '0x88c2c450571ec37d:0xacc1b92ce343345a', 
                         '0x88e7e6e8f94d5477:0x5ea802f638c2c10c', '0x88e8bb2c9331b017:0x841313652f46b77d', '0x88e77f0a3d7f9003:0xa1f36683998981be', 
                         '0x88d9aea9e5294395:0xa7c7d1e99e52ee4', '0x88c28e1e32f25c11:0x9ab145f031a57c9b', '0x88e455962cdeb27d:0x13438f2a3b997544', 
                         '0x88c2c429bdb4457b:0x61f8af82691f5546', '0x88e66594591416c7:0x69d77d4b9e935680', '0x88d9bc8a83bc0011:0xe219239f2b7508be', 
                         '0x88d9bb2fc75a46ef:0xf8003f95d0b7e8eb', '0x88d9bb073fddb393:0x6bbe2f5ff959daed', '0x88d91d192f78711f:0xe43a6e1f15bdb67b', 
                         '0x88e8bc50dbd4ef89:0x2430eefb445dfc4c', '0x88c2cd00fe87b5db:0xd1fcd4c623cda026', '0x88deea5508a742e3:0x57a71f41b71d5930', 
                         '0x88d9c3894c9e4a6d:0xad9edf11e7eb843f', '0x88deeb1f017303a9:0xe1533f7a1dce1706', '0x88d9a438b918661d:0x3d8d056532cae074', 
                         '0x88e8bb764b40114f:0xdcdb306193107b23', '0x88913e5faf56c9c9:0x59cb87e3bd83b59d', '0x88d8dfdb2fda46c9:0xea4fe0eb8d234962', 
                         '0x88dd7ca2cb488799:0x702e9894b76261db', '0x88d9b7be2bc6af9b:0x84767e6c9145a155', '0x88c34184f8865b43:0xe1366d7d7bb0cad7', 
                         '0x88e7c40cffa83a3d:0xb86d06f50ef79a35', '0x88de0e0c720b256b:0x64fd62a119934611', '0x88c2ed0a25ab9b47:0xf3255b4ab036903b', 
                         '0x88c2fa7a115bc43f:0x965cd51456de404', '0x88d9c6f3ca83f091:0x98cb164ca179e4c3', '0x8890b9489881e63f:0x3726e19879bf864b', 
                         '0x88e64580135490c5:0x6a4d5763d177110e', '0x88e8beb88fc3d07b:0xb28fcb025500cc53', '0x88d91e042ec48fe5:0xb3e93e2274f720f4', 
                         '0x88de11e56132dd71:0x12744f9d81824242', '0x88e8a384848aaef1:0x5f476b5e372addaf', '0x88d9050d9453e975:0x35e11d307f895db2', 
                         '0x88d8d7daab0ec517:0x55df53b8b6e230a8', '0x88d9abcf45918cb3:0x7a07c4f1ea53561e', '0x88e76d198484e8c5:0x266db51127e50eb7', 
                         '0x88d9b3d53e506261:0x50727d55496e2d87', '0x88d9af3b95eada5b:0x5e955c23bf41364d', '0x88d92816ea1cf967:0xba895b391059d8f7', 
                         '0x88d9bc060b97f3b7:0x39c6b30c913c0f1c', '0x88c2c2fb7e6a84c5:0x927248de1f4ee114', '0x88d9bb229ebd5b39:0x18751c638e243ea8', 
                         '0x88e5b357ad2bffd7:0x68e75dc13d9f2310', '0x88dae2eda193b7c7:0x8a1a5b473391c144', '0x88d9287d8f15dce7:0x1e0e1e82ee49aa18', 
                         '0x88ded537f37caa25:0x563081785dc12f43', '0x88d9a949d4240585:0x1e62cc948d859dbe', '0x88c2f28c182ec77d:0xb9351091fdf85ddd', 
                         '0x88c2ed2a53903897:0x10ff67e6eadb5f97', '0x88d9afb8402d1cb5:0xb4493ccfab234634', '0x88dd3b9594a0a5d3:0x53d637fbcca8e88f', 
                         '0x88d9ac3dc72ced45:0xfc742c7ca7e5b77b', '0x88e5b0d3be494a1f:0xa29cefbcc13f5a7', '0x88e7ce3b7ca856e5:0xc68d4e7c31ec64bc', 
                         '0x88dc5588408a2179:0xaff9d082b4135a13', '0x88d9a8546e3fe59f:0xf11858c2e59667ad', '0x88e7c355bc43ebc5:0x5e17f2f9676c6fd0', 
                         '0x88d91d2eaced4f35:0xa3c6aa03d70e1f99', '0x88e5e791c183fe79:0x91aaa2a81e54ea4f', '0x88d9b252b4f9ba73:0xb76b7b90389f5d72', 
                         '0x88e5cb76eebb6af9:0xdde753b37fe437e3', '0x88db3c74c7c303cd:0x8cd8ff6923898eea', '0x88e8705c4070d2e5:0x1f0b1ae7f8316d29', 
                         '0x88c2c3b6f7bc86f7:0xdbb0e2437843dae6', '0x88d9ca72eb2926cb:0x74e712c0719db790', '0x88d90510072c0c95:0xcde1d475db1a0bca', 
                         '0x88c2cdac429efa89:0x369c47f2154b33f7', '0x88c2f3b28c83d6ad:0x3cda60d2996c7d0e', '0x88db16c805542897:0xb8b95e4ffdd07e77', 
                         '0x88dd3c3a466aaa55:0x44101faf78205c2d', '0x88db3d054c1e8b81:0x86769a97ee6eb800', '0x88c2fb287d68276d:0x284f1b2dd373e25f', 
                         '0x88e44e8a53f08a73:0x989ddd7be529595e', '0x88db498ffa2a4b83:0x225fdec26a4ec8fb', '0x88d9abf94de50ff1:0xc50838bae5fc1704', 
                         '0x88e6dafbe24655a9:0x1cd2d59e3c24e41e', '0x88d91c7df99a1ca5:0x7f3104f088cc3859', '0x88db1543d055741b:0xda598599b260b5d4', 
                         '0x88e7cd914afd7eb7:0x5fbb52588f9faea1', '0x88c2c317e5f3fd71:0xdf909fe13fde6a03', '0x88c2dcf5419e9ff3:0x7c351b3069ef7a35', 
                         '0x88c2ea9105c21097:0xc2cf464b13cea08b', '0x88d8e1c12fdda271:0xc0e05f7ba766a76f', '0x88d9e0b5ca62c45d:0x6aeb1cc294c6e314', 
                         '0x88d90a69f1c38f3b:0xa03f37d07acbb3b4', '0x88d909b7d7e779cb:0x8c34ccf38645c73f', '0x88de61a5861f9a6d:0x4ace2de1a8ceb041', 
                         '0x88ec6086f5083695:0x4de0a7d7b264525d', '0x88e785ad0a49c843:0x36e951df107ffdfb', '0x88d9bb4f47e2e9b3:0xae5fcc1d3d17fbec', 
                         '0x88d9b6b783cdf50d:0xb3c46674bb96bcf4', '0x88e5f311555a3b65:0x472ecf94a1604a48', '0x88e8a4631d4b19f3:0xeee5d23e522e5f22', 
                         '0x88c3168ed2fd231b:0x98ad3a1508f6253f', '0x8890c120ef89d175:0xe73e886c1c233a24', '0x88dd12f70a94a0a5:0xf382897861f241e4', 
                         '0x88d8d8f77f50dd27:0x48ee521492edd9d2', '0x88e74a7741f45ce1:0x16863981b3496b47', '0x88e63a09070d6b73:0x7140c44af6ebb365', 
                         '0x88d91f792fc860eb:0xd2fb48ed9581279c', '0x88d9c3dfd1b2edc9:0x3b99a1bd5d1c73ce', '0x88c4a82ec45a1003:0x37095b6aaf20e567', 
                         '0x88e7114d34dd0ed5:0xa8a76a96f7f2cdcc', '0x88e0aa4fc137ce8f:0xca2debe40562cd21', '0x88e0aa6cdf858b07:0x5b803879f9dae14e', 
                         '0x88de607fcc902af7:0x7831a570c46319c6', '0x88def190a2df1bb1:0x56be2fba19cfc887', '0x88e8bd120be1e901:0xdf932b4b959d9d59', 
                         '0x88d9ab981a112211:0x60dbc9e5f10a2c7c', '0x88daffe9d62a7feb:0x32e23a846fa4389a', '0x88e841b0afac32f1:0x7224ca8b18fc88c6', 
                         '0x88dae4b82c5c94cf:0x9ebfed849ff0993e', '0x88db6dbd8497537d:0xc1900edb02413876', '0x88d908f0d77eda9f:0x48ff398e4e15c0b6', 
                         '0x88d9b86c4752623f:0xd044aa3e8ca96eca', '0x88dd87601e2b8895:0xeb6e4d4c53f5b7b5', '0x88c2f6d145b69727:0xb23a15d8600c7d9a', 
                         '0x88d9a7de8ee7d6eb:0x319731f5e7caa546', '0x88c4a9b687ed84a3:0x9109105bfe13181f', '0x88c359e5502634a1:0x9bffed81910cf382', 
                         '0x88d9c121706af551:0xa5feda8e529ad02', '0x88db6ba6777e0c63:0x74064a02d627766b', '0x88e77ebb789a924d:0x7a9adda1089b1351', 
                         '0x88e7be72146ad1cb:0x2da799273c81e596', '0x88d9bea15500770b:0xf2e21ab061dea46e', '0x88e71af6f289582d:0x91189b59ea4538aa', 
                         '0x88e7793a94847cb7:0x57e1ace75df70cf3', '0x88e7704a4fd8381f:0xfc90e24302b6676e', '0x889378e010cc2ee9:0x4e490b73df1c3735', 
                         '0x88c2ee26e10ed0db:0x8089fa4aed871894', '0x88db3ef15291be3f:0xa951f4a57b5419da', '0x88dd394b89e66979:0xab3cf53b5aabc832', 
                         '0x88de72ec67d3dbbf:0x772861a77f2bd206', '0x88e7643e7d39693b:0x4234e9205ac7f60e', '0x88e5c54101d40659:0xe9abcf4a1c91e35c', 
                         '0x88e65e7243bf980f:0xfe1e15ff31727efe', '0x88d8d6565e8e708d:0xaa47c36f9b557626', '0x88d9c0acb07f2625:0xfdcc69215223dcf2', 
                         '0x88db41e84aa2d2eb:0x96a18fe732c8d10a', '0x88d9bc88c9e6fe03:0xff5e00d872d0bfc6', '0x889395942b187683:0x678cdbf0bc0abb93', 
                         '0x88c4a98fb3b82079:0xc25ee43616a6d41a', '0x88d9a9e2f5039209:0x8f17da398921d1da', '0x88db1ca07e9dfb0d:0x2388e39ca12f7a95', 
                         '0x88e79de0145c2bcf:0x36bafcd56d105335', '0x88d926fe62c57f01:0x45e6954f950d0a38', '0x88e788d380b2303d:0x557eb6c09ebcc3be', 
                         '0x88e8a384848aaef1:0xf5116fb065d324ea', '0x88e93dc67f26f251:0x7e01fb6ecf3cc99a', '0x88d9b16b69ffa347:0x12dc225b65aa3a1f', '0x88dc2abcc35b6a35:0xd2b55f38388b9b4a', '0x88deebea0a28b8a1:0x2d731af88580cef3', '0x88d9b1a39771956b:0xe1ae22bb5d94ddfd', '0x88db401a9dab077b:0x24091c799e58f2a1', '0x88c2c01253b887a1:0xb8f7c451bc293ad0', '0x88c2bf0fdbc455e7:0xa1a280ad50069934', '0x8890c0bea98ae17f:0xc354c5244c92fa28', '0x88e5c67d3659fdd3:0xdf67c4fedda31137', 
                         '0x88e8796ce9c8fdd3:0xa2b81bba4c3817d0', '0x88d9ae8df3432f35:0xe0a0f636b7dc7b03', '0x88db4006b0e97493:0x87aad33f7158d3f4', '0x88d9a9469a65e8e5:0xeba6fa922a03d5ac', '0x8890eda9addc5aa7:0xbd0b7aa48d058c04', '0x88ecf519ffcc7efb:0x5c82d78c8f5819fa', '0x88d9a97ea515ba4b:0x98dcc06d5beacff6', '0x88dd3eadff688165:0xf668ec4e8bedd169', '0x88e5e60450178207:0xde657c0f0184d419', '0x88de0501b99ddd3f:0x1819552c65b9d8a6', 
                         '0x88e7697792da1613:0xf18f3cbb5d660e8a', '0x88e6c4b8b3b94939:0x5d3cd57de2d22b9d', '0x88e60d5b8a76c313:0x57561dcf135a5146', 
                         '0x88c2c37b07ffa271:0x31ae6d85051273e3', '0x88d9a56bed720047:0x3deead66272581b1', '0x88d9b7b3d1a6cd5d:0xe2418cb0c3faa789', '0x88e70d4037473405:0x3260a3644c4fcf2e', '0x889388c574244ef5:0x8bcebc1e82f0c2e8', '0x88e77850a5e0d5f7:0x9f88ac1c6b3e62b', '0x88e7685c30e1a7ad:0xb2cb7fd4d963abe6', '0x88d92a6d72b5b953:0x8f68e2a8b7aef0dc', '0x88c29bc621d0af53:0xbfccf43983706ea5', '0x88dc5588408a2061:0xbd8e9fafb6bd018b', 
                         '0x88dbfa52016357b5:0x62be0ef5026a3c8d', '0x88d9b0260571e1bf:0x7798afd3696df750', '0x88d9a7a4c0cef387:0x5525e780ca46c011', 
                         '0x88d95a83c4340123:0x5059eaf8060ca775', '0x88d9015d3cbcb053:0x3d347982c66c5266', '0x88d9e7de085641b7:0xc9407a848f127dd6', 
                         '0x88d92f27df0899ff:0x578870c21de87de6', '0x88e8dde8a01065af:0x7bad81cfe7341b3b', '0x88d8d689d6394307:0xa5c7868dfad0f99c', '0x88e687f0bef4b43d:0x39e341aa7b4ccf51', '0x8893d1ad8c8f569f:0x2dc15b0b9d53462f', '0x88d906bc0fc5a4bf:0x2c20b39d28012bbd', '0x8893e514b7ee104d:0x5a07b2d0ac9d2d08', '0x88def100640e75dd:0x8897f79747048acd', '0x88e5933dc933b4a1:0x3ef6a7bd402dd0a', '0x8890e8c8a69f46d9:0xf4aabd14250216c3', '0x88d9030f52a8e301:0xbcb1af4dc6bf9eb1',
                         '0x889169ad9bffc287:0x7b2b8198f6d3d496', '0x88e5ae4e2de56a95:0xf7a636a26f386abc', '0x88e719fd128d39e5:0xd35947e705276b0a', '0x88dd73ca185957a7:0x594d2beb78d3e3b9', '0x88d9bb685bedaed7:0xfbebd183f58dffcd', '0x88d9bcba82460b9b:0x18ea3ca75e93db98', '0x88d9021726bdc5d1:0xa431357c3fedac84', '0x8890c027f074ba45:0x6c38dec2949b3011', '0x88e66443ee8b8ee7:0x9e662cc423889ef3', '0x88c2e32e8941d18f:0x93090baa85fda043', 
                         '0x88d90f9e0a054a01:0x74f106a4271a77b0', '0x88de5fd487c8ca27:0x476cfdbb24a9ffa', '0x88dee7d652fc920f:0xb723833b378ba510', '0x88daefaea90a5e43:0x31abb1a3cf169da4', '0x88c3228b5aefc66d:0x2727d495cbf1f22c', '0x88c2fa9932c0a6e1:0xa1736afcb8db6f8d', '0x88e696650de41235:0x3774bcb1c2cfdc12', '0x88e5b620b01d0123:0x36a6ec49ba573a0c', '0x88e798dc840be9d7:0x79cc9d139cd1f556', '0x88e7a257060a7993:0x8bc7e2d3bfb4e7ee', 
                         '0x88db726a216fce05:0x1fb1f0d8e0ad58e9', '0x88e73df544cf1a4b:0xa105fa25db98db57', '0x88db4147f24e56b7:0x25089a10d3f848dc', 
                         '0x88e7dd2e3b986eb9:0x29fef68dcb2848f8', '0x88c35486ee70f4cb:0xb1f30c34bb6553aa', '0x88d740c07b6bb605:0xf48faa1f35de66c3', '0x88d9b7a540e6dc0b:0x335a8eaed4a5ab32', '0x88d9009af50062e9:0xe8c36d74bb3f2fb7', '0x88d920fd4f66e8cb:0xd049b659155543c3', '0x88e5a82c2c0d7035:0x7b6306cf8311c88b', '0x88e69d469e618b0b:0x5ce04808db861555', '0x88e44ae595e7e0a1:0x6b1dfbc94640795', '0x88c2cc0205ca0737:0x88276f62eca8b6c3', 
                         '0x88db69147a3f1693:0xbe9c445167ee3851', '0x88d8df59a4a64d3b:0x5d682e5b1490c4d7', '0x88d9bacdc7b2ba69:0x58bbc63ba3fb3a46',
                         '0x88d9a82c728f51df:0x94afff31428d0e8a', '0x88d1b15cd5455107:0xfa4d20e35e1687fe', '0x88d9b185fc05d283:0x84f50e170244c394', '0x88ef3bd5da2d59f5:0xbe7806290d1b3dbd', '0x88d9ab3ca6488ef9:0xf4c6118da6878d3d', '0x88e5c19c5c2d89a9:0x3cff1e4d39149810', '0x88db3e585af9f487:0xa4081b036a87d5a3', '0x88deee65f30df27d:0xa609b582d26b76a3', 
                         '0x88d9a5d74bc39f1f:0xc756fda528fc13ea', '0x88d904077d91b73f:0x9e5d365cee837d48', '0x88dd011729140d05:0x94cb56a18758336a', '0x88c340ea86ee389f:0xdb6a82e135adf28', '0x88dc59e1bfd3c4df:0xe4a88583aef7ce7d', '0x88deebfbb944c253:0x37a5935fb0dd4816', '0x88c2c4c085dc029d:0x2a1b5389c40bf3d7', '0x88d9b69deaabd6d1:0x9c31e5226604ba42', '0x88d9c243b5382559:0x87f577e3e37759e0', '0x88c2f083016e6fb5:0x762ed732b6df2a4f', '0x88d9b6a052763061:0xc386ff5aa9504d36', 
                         '0x88d9c22959c3e7c9:0xda59eba74666b109', '0x88d9c39af294fac1:0x69cdd30045a30481', '0x8890badfd13b5b03:0x8dc953d0a385ff63', '0x88c2c3f2399b17f7:0xe92f2aaf1d1fe5f7', 
                         '0x88dc57c81980658b:0xd081a42b950be3ad', '0x88e77ef1d23c5007:0xcb2042a6714fae50', '0x88dafccbeadfeb09:0xdc7822300943d7a9', '0x88e574532246e073:0xa8fc31fc13c0b3aa', 
                         '0x88d91e2636d75891:0x3293be1bc792f2b3', '0x88e6945d89a91bbb:0xc5e31bd9b1033f4c', '0x88e7639adfbecd6f:0x31a03f4aa77c66bc', '0x88c2c9581ca6bdcf:0x2822d41a3da8e33', 
                         '0x88d9ac56937054dd:0x5bc78b4a15a3a914', '0x88e5c319ccbd0ff5:0x2ad1578a12c5bd5a', '0x88d92aa6cdc76dfd:0x591718b6393b9075', '0x8890bb7f4a30c5f5:0xec1090b814a9a354', '0x88d9b6a79b332afd:0xde94485fe335a941', '0x88e7962526a8f1cb:0xa72bbfa4ea67c8a6', '0x88db6a1c58971539:0x132db4813d8c5deb', '0x88d9c37ea9ff58d5:0xf07387742fe0cf20', 
                         '0x88df282a1a99586f:0x927c7abb72b37bfa', '0x88c2976830f56185:0x4401f84e6aa63a14', '0x88c2e554256420d1:0x38a72eb8df992b5a', 
                         '0x88e78cd4764af1b3:0xc255cce55d0f2e5b', '0x88914478b08ccd31:0x445347276b61cf82', '0x8890a57efffc1ac9:0xa99c50dc34fa64a', 
                         '0x88d926d3c0e6af49:0x9e535f2b0252eb20', '0x88e7d39efd1eda5f:0x665a7c05f2c0a8f0', '0x88db1c77f511a6d3:0x2c737fa8aeb870fc', 
                         '0x88eb6651c627358f:0xfec6f85a3af42823', '0x88d91f914bc68767:0x9074c5ba61223db9', '0x88e62ec5909a2061:0x89d9825e4001e9f5', 
                         '0x88e5ae28eeb76841:0xfd731d269a0c97c7', '0x88909404b2aba4b9:0x4f31b4e257466866', '0x88c2c3d46cc89cb9:0xa71cabefee526ffe', 
                         '0x88e449a0d4cc67ab:0x25e2491dba6f3c95', '0x8890c03dc0914ce9:0x30499f8bb06a7fc9', '0x88d92790c5b85d45:0xa58fbe5199c1de81', 
                         '0x88e8a39e42a9d089:0xb3a279400af50ed6', '0x88e89eddf461b95d:0x73c38175ac394635', '0x88e8a2c359bb13d9:0xcb4e08db278c42de', 
                         '0x88e61a9f09906d59:0x1957014a37199d9b', '0x88dd86587d3d407f:0xd4d6922660563a5b', '0x88dee7063e758dd7:0x8767fc68bb9f155', 
                         '0x88d9c37ea9ff58d5:0x48704d8cf6721d2c', '0x88c2ea6010ddc917:0xb1b168281c906905', '0x88db40c631970e75:0x9ef02a612374d68c', 
                         '0x88e6936558c2ece3:0x39393ae2b42b0b60', '0x88d91cca56da5bed:0x97a5f8aae02fb6d3', '0x88e6db677d81274f:0x65166bfba600c609', 
                         '0x88e74d7e0d2839e3:0xb00491792683954', '0x88905ea8ad151e37:0x71852923e1ede2ec', '0x88d9a53b46a091ed:0xd14c12e69fab13cf', 
                         '0x8890c0ac28b6da3d:0x4596cb88698ecda0', '0x88db6fba74be4857:0xec4ad354bcfb2cab', '0x88d9b7a34532d1fd:0x9855d6ca0e911f7a', 
                         '0x88db57f04586fb45:0x886723827cab7a92', '0x88d9ba8d27df1393:0x6e0f719436a6fdf5', '0x88e81cbc9c113cf9:0x303a0830213c85c7', 
                         '0x88e81feaca754185:0xc78e6f1c2be57299', '0x8890c4b23bf1a2ab:0xd8aacb09f63a40a0', '0x88e7c3d9bd54bb03:0x8bfed5724f9d65df', 
                         '0x88d9dd02f12eff8b:0xbf2a0a8b74e47ec8', '0x88c311b149673bd3:0x2bd932c785bb6232', '0x88e0a9e0b9f8e661:0x788426bce42cea6e', 
                         '0x88dd73ce3d18370b:0x8dbc297eedfba8b1', '0x88c33a554b37c4bb:0x638fc92330a321a5', '0x88e782a1a6b38d8b:0xf1d2b804bba71f4a', 
                         '0x88dec2792c49ad33:0x66ba8505e31bc709', '0x88d9b81a02727839:0x5432619e06d009cf', '0x88d9b151513ebf37:0xa836967724285034', 
                         '0x88dee8ed624a41a5:0x92d57c8d87ced598', '0x88deeb87ee19672f:0xef0f3239fbdd0ac2', '0x88e775f408bf8043:0x85ee709834d202b', 
                         '0x88e76bdf593a4e03:0x430f96a574914004', '0x88e7755de6359e13:0x12dedc3b08a0e206', '0x88e8a438cca01c7d:0x517e6e404b67f445', 
                         '0x88c2e52a06d0ee0b:0xd5e6881421b9f64', '0x88ec59c6fee87377:0xcd3946ef0433316b', '0x88c2c8aad8ac990d:0x48bbea9847f091cc', '0x88e6d8f451e87ecd:0xa03b26411c0494a4', 
                         '0x88d9c7e51cc8f8b3:0x809ade8bf2b503', '0x88e77f2109a768fd:0xd8d59db92fe8ff68', '0x88e712692d8ea761:0x8491f6472ca64d83', 
                         '0x88dae3c631eeba51:0x66e8927c88bb4c7a', '0x88dd7fa2f09daa79:0x72d5fa6f365d5335', '0x88e42d57522a8a5b:0xcebf9311703a583']        
        google_review_florida = df[df['gmap_id'].isin(lista_locales_florida)]
        # Eliminar las columnas que no seran utilizadas para el analisis
        google_review_florida = google_review_florida.drop(columns=['pics','resp'])        
        # Elimina filas completas duplicadas
        google_review_florida = google_review_florida.drop_duplicates(keep='first')
        # Convertir la columna 'time' a formato legible
        google_review_florida['time'] = pd.to_datetime(google_review_florida['time'], unit='ms')
        # Crear columna con el año de la columna 'time' 
        google_review_florida['year'] = google_review_florida['time'].dt.year 
        # Extraer solo la fecha sin hora
        google_review_florida['time'] = google_review_florida['time'].dt.date
        ## Convertir los tipos de datos
        google_review_florida['time'] = pd.to_datetime(google_review_florida['time'])
        google_review_florida['year'] = google_review_florida['year'].astype('Int64')
        #Filtramos solo reviews desde el año 2015
        df_final = google_review_florida.loc[google_review_florida['year'] >= 2015]
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
        # Crea la nueva columna 'state' con el valor 'Florida'
        df_final['state'] = 'Florida'
        df_final['short_state'] = 'FL'

        #=============================================================
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
            if main_folder == "google_review_florida": # Nombre carpeta dentro del bucket 
                
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