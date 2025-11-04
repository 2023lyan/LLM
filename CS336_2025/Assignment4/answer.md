# Problem: Look at CC
(a)
- The URL is http://0371rykj.com/ipfhsb/34.html, which comes from the WARC-Target-URI
- It's not accessible now
- It's a Chinese AV website corresponds to a Chinese webpage whose HTML title and meta tags contain encoded phrases such as "人妻国内熟妇熟女", indicating adult content.

(b)
- The extracted WET text contains a mix of explicit adult content (e.g., "人妻、国内老熟女、亚洲AV无码") and unrelated commercial information such as "Shanghai Linpin Instrument Stock Co Ltd" and product lists.
- This shows that the text extractor failed to separate the main page content from embedded advertisements or template elements.
- Much of the output, including pinyin annotations in parentheses, is clearly noise and should have been filtered out.
- If such noisy and NSFW text were used for language model training, it could lead to unsafe generations and degrade model quality.
- The only potentially useful signal is that the language was correctly identified as Chinese, which could still be used to test the language identification component.

(c)
- For example, in a content-moderation or NSFW-classification task, this document would be a valuable **positive sample**, since it clearly contains explicit language and adult material.

- However, for general-purpose language modeling or instruction-following datasets, this text would be low-quality and undesirable, as it contains explicit sexual content, mixed encodings, and irrelevant boilerplate information.

(d)
|  #  | Language | Domain        | Type of Page         | Notes                                         |
| :-: | :------- | :------------ | :------------------- | :-------------------------------------------- |
|  1  | zh       | 0371rykj.com  | Adult Site           | Low quality |
|  2  | zh       | 10www.chinatikfans.com  | Blog Page  | Low quality |
|  3  | eng      | 13.usnccm.org | Commercial Ad | Low quality |
|  4  | zh       | 176.utchat888.com | Adult Site     | Low quality |
|  5  | zh       | 178mh.com  | Adult Site          | No template |
|  6  | zh       | 1796370.tgtg97.com | Adult Site          | Low quality |
|  7  | zh       | 18sex.v340.info   | Adult Site          | Low quality |
|  8  | nld       | 1kb.klimtoren.be   | Blog Page         | Low quality |
|  9  | ell       | 1pekesat-exae.mysch.gr  | Search Page          | Medium quality |
|  10  | ell       | 1pekesat-exae.mysch.gr   | Search Page         | Medium quality |
|  11 | zh       | 1s6605084.yhxzseo.com   | News | Medium quality |
|  12 | tur,dan  | 20com20.fr   | Technical Documentation          | **High quality** |
|  13 | eng      | 24ktcasino.net | Gambling News  | Medium quality |
|  14 | eng      | 2kgames.eu | None          | 404 Not Found |
|  15 | zh       | 2l6185919.yizhangting.com   | Gambling News  | Medium quality |
|  16 | zh       | 303323.com  | Corporate Site          | Medium quality |
|  17 | zh       | 30bad.com   | Comic Site          | Medium quality |
|  18 | zh       | 312001.net  | Hospital Menu   | Low quality |
|  19 | zh       | 354577.mwe075.com   | Adult Site          | Low quality |
|  20 | eng      | 356.schoollibrary.edu.pe.ca   | Library Search Result Page  | Low quality |
|  21 | zh       | 366392.haaxz.com | Adult Site          | Low quality |
|  22 | zh       | 366392.haaxz.com   | Adult Site          | Low quality |
|  23 | zh       | 387tel.com   | Adult Site          | Low quality |
|  24 | spa       | 3diasdemarzo.blogspot.com   | News | **High quality** |
|  25 | dan       | 3godetilbud.dk | Commercial Ad | Low quality |

Across 25 samples, most webpages were low-quality (adult content, ads, or navigation templates).

The account of the webpages with high quality is close to $\frac{1}{12}$.

# Problem: Extract Text

- Our extractor produced cleaner and more readable text, successfully removing HTML tags, scripts, and boilerplate, while preserving the paragraph structure.
- Our version sometimes introduced more blank lines — this occurs because the resiliparse extractor preserves paragraph and block separation from the original HTML, simulating the layout structure.

# Problem: Language Identification
(b)

- Errors in the language identification step can therefore lead to mis-filtering — for instance, discarding English data or including pages written in the wrong language. Mixed-language or short documents often yield low confidence scores and may be mislabeled.
- In high-stakes applications, these issues can be mitigated by applying a confidence threshold (e.g., 0.8), using ensemble classifiers, or manually inspecting low-confidence samples.

(c)

1: label: zh, conf: 0.7369957566261292

2: label: zh, conf: 0.923484206199646

3: label: en, conf: 0.8091545701026917

4: label: zh, conf: 0.9952414035797119

5: label: zh, conf: 0.9251600503921509

6: label: zh, conf: 0.9060114026069641

7: label: zh, conf: 0.9581619501113892

8: label: zh, conf: 0.9791116118431091

9: label: nl, conf: 0.9234433770179749

10: label: el, conf: 0.9998586773872375

11: label: el, conf: 0.999869167804718

12: label: zh, conf: 0.9810582399368286

13: label: tr, conf: 0.8719401359558105

14: label: en, conf: 0.9264044165611267

15: label: da, conf: 0.28138768672943115 (Error)

16: label: zh, conf: 0.9614605903625488

17: label: zh, conf: 0.9708634614944458

18: label: zh, conf: 0.9297752976417542

19: label: zh, conf: 0.9965465664863586

20: label: zh, conf: 0.9558992981910706

fraction of documents are English: 10%

I think 0.8 is a suitable therehold

# Problem: Mask PII
4. Naively applying PII filters can cause important context loss. For example, masking all numbers might remove dates, measurements, or version numbers that are semantically meaningful. Over-aggressive regex patterns may also over-mask text, introducing noise into model training. To mitigate this, filters can be refined with stricter context checks and evaluated on validation samples to balance privacy protection and data utility.

5. No false positives or false negatives situation

# Problem: Harmful Content
3. Naively filtering out all text flagged as harmful may remove important linguistic diversity or legitimate discussions that contain certain keywords (e.g., news articles about hate speech or sexual education). This can bias the training distribution and cause the model to underperform on sensitive or nuanced topics. Over-filtering also risks reinforcing cultural or social biases embedded in the classifier. To mitigate these issues, filters should be applied conservatively, combined with human review or multiple classifiers, and regularly audited for bias and coverage.

4. When running the harmful content filters, the classifiers successfully detected NSFW or toxic content in English pages but failed to identify similar content in Chinese websites. This is because the fastText models were trained only on English Wikipedia comments, so they lack coverage for multilingual or domain-specific data. As a result, many pornographic or abusive Chinese pages were labeled as “non-toxic” with very high confidence. To mitigate this issue, multilingual fine-tuning or additional training on diverse web data should be performed to improve cross-lingual robustness.

# Problem: Gopher Quality Filters
(b)
Results:

page: 3
Low quality
page: 14
High quality
page: 21
Low quality
page: 27
Low quality
page: 29
Low quality
page: 52
Low quality
page: 53
High quality
page: 55
Low quality
page: 59
High quality
page: 68
Low quality
page: 81
Low quality
page: 83
Low quality
page: 87
High quality
page: 88
Low quality
page: 89
Low quality
page: 91
Low quality
page: 92
Low quality
page: 97
Low quality
page: 103
Low quality
page: 107
Low quality

The prediction is very close to my own judgement

# Problem: Exact Deduplication

