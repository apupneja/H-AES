import transformers

prompt_1 = '''अधिक से अधिक लोग कंप्यूटर का उपयोग करते हैं, लेकिन हर कोई इस बात से सहमत नहीं है कि इससे समाज को लाभ होता है। सकारात्मक प्रभाव यह हो सकता है कि वे पढ़ाते हैं, लोगों को दूर के लोगों के बारे में जानने की क्षमता देते हैं, और यहां तक कि अन्य लोगों के साथ ऑनलाइन बात भी करते हैं। इस बात की चिंता है कि लोग अपने कंप्यूटर पर बहुत अधिक समय व्यतीत कर रहे हैं और व्यायाम करने, प्रकृति का आनंद लेने में कम समय व्यतीत कर रहे हैं। अपने स्थानीय समाचार पत्र को एक पत्र लिखिए जिसमें आप यह बताते हैं कि कंप्यूटर का लोगों पर क्या प्रभाव पड़ता है।'''  # 114 Words
prompt_2 = '''पुस्तकालयों में सेंसरशिप पर आपके दृष्टिकोण को दर्शाते हुए एक समाचार पत्र के लिए एक प्रेरक निबंध लिखें। क्या आप मानते हैं कि कुछ सामग्री, जैसे किताबें, संगीत, चलचित्र, पत्रिकाएँ आदि, यदि आपत्तिजनक पाई जाती हैं, तो उन्हें अलमारियों से हटा दिया जाना चाहिए? अपने स्वयं के अनुभव, टिप्पणियों और/या पढ़ने के ठोस तर्कों के साथ अपनी स्थिति का समर्थन करें।'''  # 61 words
prompt_3 = '''एक प्रतिक्रिया लिखें जो बताती है कि सेटिंग की विशेषताएं साइकिल चालक को कैसे प्रभावित करती हैं। अपनी प्रतिक्रिया में, निबंध से उदाहरण शामिल करें जो आपके निष्कर्ष का समर्थन करते हैं।'''  # 99 words
prompt_4 = '''कहानी का अंतिम पैराग्राफ पढ़ें। "जब वे वापस आते हैं, तो सेंग ने चुपचाप अपने आप से शपथ ली, वसंत ऋतु में, जब बर्फ पिघल जाएगी और हंस वापस आ जाएंगे और यह हिबिस्कस उभर रहा है, तो मैं फिर से वह परीक्षा दूंगा।"एक प्रतिक्रिया लिखें जो बताती है कि लेखक इस अनुच्छेद के साथ कहानी का समापन क्यों करता है। अपने जवाब में, कहानी से विवरण और उदाहरण शामिल करें जो आपके विचारों का समर्थन करते हैं। '''
prompt_5 = '''संस्मरण में लेखक द्वारा बनाई गई मनोदशा का वर्णन करें। संस्मरण से प्रासंगिक और विशिष्ट जानकारी के साथ अपने उत्तर का समर्थन करें।'''  # 112 words
prompt_6 = ''' अंश के आधार पर, एम्पायर स्टेट बिल्डिंग के बिल्डरों को वहां डॉक करने की अनुमति देने के प्रयास में आने वाली बाधाओं का वर्णन करें। अंश से प्रासंगिक और विशिष्ट जानकारी के साथ अपने उत्तर का समर्थन करें। '''
travel = '''यात्रा, हमें हमारे आराम क्षेत्र से बाहर ले जाती है और हमें नयी चीजों को देखने, स्वाद लेने और नयी-नयी कोशिशें करने के लिए प्रेरित करती है। यह हमें लगातार नए वातावरण के साथ समायोजन करने, नयी खोज करने, भिन्न-भिन्न लोगों के साथ जुड़ने, साहसिक कार्य को करने, मित्रों और प्रियजनों के साथ नये व सार्थक अनुभव साझा करने की चुनौती देती है। यात्रा हमें मानवता के विषय में सिखाती और हमें भिन्न-भिन्न दृष्टिकोणों व जीवन के मार्गों के प्रति सही-गलत की समझ और सम्मान देती है। यह हमारे जीवन में सकारात्मक बदलाव के साथ-साथ हमें जीवित और सक्रिय रखती है। '''

DEVICE = "cpu"
MAX_LEN = 512
TRAIN_BATCH_SIZE = 6
VALID_BATCH_SIZE = 4
DATASET = "./dataset/travel.xlsx"
EPOCHS = 8
LR = 3e-5

indicbert = 'ai4bharat/indic-bert'
mBERT = 'bert-base-multilingual-cased'
distilmbert = 'distilbert-base-multilingual-cased'

xlmr = 'xlm-roberta-base'
muril = 'google/muril-base-cased'

BERT_PATH = mBERT
CURR_PROMPT = travel

TOKENIZER = transformers.AutoTokenizer.from_pretrained(BERT_PATH)
MODEL = transformers.AutoModel.from_pretrained(BERT_PATH)
