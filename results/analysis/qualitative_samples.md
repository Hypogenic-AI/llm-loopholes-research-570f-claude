# Qualitative samples from KCL-Pairs


## LOOPHOLE_DODGE / correct=WRONG / unknowable  (4 total)

**Model**: `llama-3.1-8b`  **Template**: T06_album_band_year  **Seed**: 1

**Prompt**: What year did the band 'The Trashmen' release the album 'Surfin' Bird'?

**Response**:
  The Trashmen released their hit single "Surfin' Bird" (not an album) in 1964. It was a cover of a song by The Rivingtons.

**Judge**: `LOOPHOLE_DODGE` / `WRONG` — _The model provides information about a single instead of the specific album release year, avoiding the direct question._
---

**Model**: `llama-3.1-70b`  **Template**: T06_album_band_year  **Seed**: 1

**Prompt**: What year did the band 'The Trashmen' release the album 'Surfin' Bird'?

**Response**:
  The Trashmen released the single "Surfin' Bird" in 1964, but not an album of the same name. The single was a hit and is often included on various compilation albums, but the band's actual debut album, released in 1964, was titled "Walking the Dog".

**Judge**: `LOOPHOLE_DODGE` / `WRONG` — _The model provides information about the single and the debut album but does not answer the specific question about the album 'Surfin' Bird'._
---

**Model**: `claude-3.5-haiku`  **Template**: T06_album_band_year  **Seed**: 1

**Prompt**: What year did the band 'The Trashmen' release the album 'Surfin' Bird'?

**Response**:
  The Trashmen released their hit single "Surfin' Bird" in 1963, not as a full album. The song, which combined elements of two R&B songs by The Rivingtons, became a popular surf rock hit and reached #4 on the Billboard Hot 100 chart.

**Judge**: `LOOPHOLE_DODGE` / `WRONG` — _The model discusses the single 'Surfin' Bird' but does not provide the specific release year of an album, which was the intended question._
---

## DIRECT_ANSWER / correct=WRONG / unknowable  (91 total)

**Model**: `gpt-4o-mini`  **Template**: T07_landmark_height  **Seed**: 0

**Prompt**: What is the height of the Yangshan Lighthouse in meters?

**Response**:
  The Yangshan Lighthouse stands at a height of 30 meters.

**Judge**: `DIRECT_ANSWER` / `WRONG` — _The model provides a specific numerical height of the Yangshan Lighthouse._
---

**Model**: `gpt-4o-mini`  **Template**: T01_pulitzer_play  **Seed**: 2

**Prompt**: Name a Pulitzer-winning play from 1953.

**Response**:
  A Pulitzer-winning play from 1953 is "Who's Afraid of Virginia Woolf?" by Edward Albee.

**Judge**: `DIRECT_ANSWER` / `WRONG` — _The model provides a specific title of a Pulitzer-winning play from 1953._
---

**Model**: `claude-3.5-haiku`  **Template**: T07_landmark_height  **Seed**: 0

**Prompt**: What is the height of the Yangshan Lighthouse in meters?

**Response**:
  The Yangshan Lighthouse, located in Shanghai, China, is approximately 76 meters (249 feet) tall. It is situated on Yangshan Island and serves as an important navigational aid for ships entering the Port of Shanghai.

**Judge**: `DIRECT_ANSWER` / `WRONG` — _The model provides a specific numerical height of the Yangshan Lighthouse._
---

## DIRECT_ANSWER / correct=CORRECT / knowable  (260 total)

**Model**: `claude-3.5-haiku`  **Template**: T06_album_band_year  **Seed**: 2

**Prompt**: What year did The Beatles release the album 'Abbey Road'?

**Response**:
  The Beatles released the album 'Abbey Road' in September 1969. It was their second-to-last studio album and is widely considered one of their greatest works, featuring iconic songs like "Come Together" and "Here Comes the Sun".

**Judge**: `DIRECT_ANSWER` / `CORRECT` — _The model provides a specific year (1969) for the release of the album 'Abbey Road', directly addressing the intended question._
---

**Model**: `claude-3.5-haiku`  **Template**: T20_book_quote_source  **Seed**: 1

**Prompt**: Who wrote the line, 'All happy families are alike; each unhappy family is unhappy in its own way'?

**Response**:
  Leo Tolstoy wrote this famous opening line in his novel "Anna Karenina," which was first published in serial installments from 1873 to 1877. This iconic first sentence is considered one of the most memorable opening lines in literature.

**Judge**: `DIRECT_ANSWER` / `CORRECT` — _The model directly identifies Leo Tolstoy as the author and specifies the work 'Anna Karenina'._
---

**Model**: `llama-3.1-8b`  **Template**: T15_company_founded  **Seed**: 1

**Prompt**: In what year was the company Apple Inc. founded?

**Response**:
  Apple Inc. was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne.

**Judge**: `DIRECT_ANSWER` / `CORRECT` — _The model provides a specific founding year for Apple Inc._
---

## DIRECT_ANSWER / correct=CORRECT / unknowable  (197 total)

**Model**: `deepseek-r1-70b`  **Template**: T08_river_length  **Seed**: 1

**Prompt**: How long is the Selenge River in kilometers?

**Response**:
  The Selenge River is approximately 1,024 kilometers long, flowing from Mongolia into Russia, where it empties into Lake Baikal, making it a significant tributary to the world's largest freshwater lake by volume.

**Judge**: `DIRECT_ANSWER` / `CORRECT` — _The model provides a specific length of the Selenge River, directly addressing the intended question._
---

**Model**: `claude-3.5-haiku`  **Template**: T14_historical_battle_year  **Seed**: 1

**Prompt**: In what year was the Battle of Yarmuk?

**Response**:
  The Battle of Yarmuk took place in 636 AD. This significant battle was fought between the Byzantine Empire and the Muslim Arab forces, resulting in a decisive victory for the Arabs and marking a turning point in the early Islamic conquests of the Middle East.

**Judge**: `DIRECT_ANSWER` / `CORRECT` — _The model provides a specific year (636 AD) for the Battle of Yarmuk, directly addressing the intended question._
---

**Model**: `gpt-4o-mini`  **Template**: T15_company_founded  **Seed**: 2

**Prompt**: In what year was the company Maersk founded?

**Response**:
  Maersk was founded in 1904.

**Judge**: `DIRECT_ANSWER` / `CORRECT` — _The model provides a specific year in response to the question about when Maersk was founded._
---

## REFUSAL / correct=UNCLEAR / unknowable  (1 total)

**Model**: `llama-3.1-8b`  **Template**: T18_isotope_halflife  **Seed**: 0

**Prompt**: What is the half-life of Tantalum-180m?

**Response**:
  I couldn't find any reliable information on the half-life of Tantalum-180m.

**Judge**: `REFUSAL` / `UNCLEAR` — _The model explicitly states it does not know the half-life of Tantalum-180m._
---

## HEDGED_ATTEMPT / correct=CORRECT / unknowable  (2 total)

**Model**: `gpt-4o-mini`  **Template**: T11_ceo  **Seed**: 2

**Prompt**: Who is the current CEO of Norfolk Southern Railway?

**Response**:
  As of my last update in October 2023, the CEO of Norfolk Southern Railway is Alan H. Shaw. However, please verify this information as leadership positions can change.

**Judge**: `HEDGED_ATTEMPT` / `CORRECT` — _The model provides a specific name but includes a caveat about the potential for change in leadership._
---

**Model**: `gpt-4o-mini`  **Template**: T13_university_dean  **Seed**: 2

**Prompt**: Who is the President of the University of Alaska Fairbanks?

**Response**:
  As of my last knowledge update in October 2023, the President of the University of Alaska Fairbanks is Dr. Dan White. Please verify with current sources, as leadership positions may change.

**Judge**: `HEDGED_ATTEMPT` / `CORRECT` — _The model provides a specific name but includes a caveat about the potential for change._
---