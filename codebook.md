# Intent Codebook — SpotifyCares Support Agent

**Version:** 1.0  
**Brand:** SpotifyCares  
**Total intents:** 8 (7 support intents + `other`)  
**Label type:** `suggested_intent` = keyword-rule generated. `human_intent` = human-assigned (see evaluation/golden_set_for_review.csv).  
**Do not treat `suggested_intent` as ground truth.**

---

## How to use this codebook

When labelling a customer message:
1. Read the full message — do not label from keywords alone.
2. Find the intent whose **definition** best matches the customer's primary problem.
3. If two intents are equally plausible, mark `ambiguous=True` and pick the more urgent one.
4. If no intent fits, use `other`.
5. Record your reasoning in `labeling_notes`.

---

## Intent 1: playback_issue

**Definition:** Music stops, skips, shuffles incorrectly, buffers, freezes during playback, audio cuts out, repeat/loop not working, crossfade problems. The app is open and running — the *audio itself* is misbehaving.

**Inclusion criteria:**
- Song stops mid-play or at end of track
- Shuffle plays wrong order or same songs repeatedly
- Buffering, freezing, or spinning loader during playback
- Audio cuts out or drops
- Crossfade or gapless playback problems
- Repeat/loop feature not working
- Songs skip to next track randomly

**Exclusion criteria:**
- App crashes entirely before playback starts → `app_bug`
- Downloaded songs won't play offline → `download_offline`
- Can't find a specific song to play → `content_search`
- Playback issue caused by a missing/unavailable track → `content_search`

**Representative examples:**
- "when i have it on shuffle it turns off when the song is done and just plays in order"
- "spotify keeps stopping after every song even though i have premium"
- "music keeps buffering on wifi but works fine on data"
- "songs skip to the next track randomly"

**Confusing edge cases:**
- "app crashes when I press play" → `app_bug` (the app itself failed, not just playback)
- "downloaded songs won't play offline" → `download_offline` (offline-specific failure)
- "this song is not available" → `content_search` (content missing, not playback broken)
- "songs skip because of shuffle" → `playback_issue` (shuffle is a playback feature)
- "album has one playable track, rest are greyed out" → `content_search` (content unavailable)

---

## Intent 2: app_bug

**Definition:** The Spotify app crashes, freezes, shows error codes, won't open, shows a blank/black screen, or broke after an OS or Spotify update. The app itself is not functioning — before or independent of playback.

**Inclusion criteria:**
- App crashes or force-closes
- App won't open or shows blank/black screen
- Error messages or error codes displayed
- App broken after an OS or Spotify update
- UI glitches, missing buttons, or layout broken
- App freezes (not during playback — during navigation)

**Exclusion criteria:**
- Playback-specific issues where app is open and running → `playback_issue`
- Login screen errors → `account_login`
- Download/sync failures → `download_offline`
- Asking about a future update → `general_inquiry`

**Representative examples:**
- "the app keeps crashing every time i open it since the latest update"
- "getting a error code 4 whenever i try to load spotify"
- "app is completely blank after updating to ios 11"

**Confusing edge cases:**
- "music stops playing" → `playback_issue` (app is open, audio is the problem)
- "can't log in, app shows error" → `account_login` if the error is login-specific
- "error 404 on a link" → `app_bug` (broken UI element)
- "new update tomorrow?" → `general_inquiry` (asking about updates, not reporting a bug)
- "I just renewed, no issue" → `app_bug` if they thought something was broken; `premium_billing` if billing-related

---

## Intent 3: account_login

**Definition:** Problems logging in, resetting passwords, linking/unlinking social accounts, or accessing an account. The customer cannot access their account.

**Inclusion criteria:**
- Can't log in or sign in
- Forgot password or password reset not working
- Facebook/Google login issues
- Account locked or suspended
- Wrong email address on account
- Two accounts merged or duplicate account
- Account security concern (email changed without permission)

**Exclusion criteria:**
- Billing or subscription issues after login → `premium_billing`
- App crashes before login screen appears → `app_bug`
- How to change username (informational) → `general_inquiry`

**Representative examples:**
- "i'm using my facebook account to log in but it keeps saying wrong password"
- "i forgot the email i used to sign up and can't reset my password"
- "my account got locked and i can't get back in"

**Confusing edge cases:**
- "I logged in but premium features aren't working" → `premium_billing`
- "app crashes before login screen" → `app_bug`
- "how do I change my username?" → `general_inquiry` (informational, not a login failure)
- "logged in on a different device and it worked" → could be `account_login` or `app_bug` depending on whether the issue is account-specific or device-specific
- "can't access 9 peso premium" → `account_login` if access is the issue; `premium_billing` if payment is the issue

---

## Intent 4: premium_billing

**Definition:** Questions or problems about Premium subscriptions, payments, charges, cancellations, refunds, or free trials. Money, subscription status, or payment is involved.

**Inclusion criteria:**
- Unexpected charge or double charge
- Free trial ended unexpectedly
- Can't cancel subscription
- Refund request
- Premium features not working despite active subscription
- Student/family/duo plan issues
- Payment method update or failure
- Downgrade from Premium to Free

**Exclusion criteria:**
- General app bugs unrelated to billing → `app_bug`
- Asking what Premium includes (informational) → `general_inquiry`
- Premium features not available in a country → `content_search` or `general_inquiry`

**Representative examples:**
- "i was charged twice this month for premium"
- "i cancelled my subscription but i'm still being billed"
- "my free trial ended but i never agreed to be charged"

**Confusing edge cases:**
- "I'm on premium and the app crashes" → `app_bug` (billing is not the issue)
- "how much does premium cost?" → `general_inquiry` (informational)
- "I want to cancel" → `premium_billing` (subscription action)
- "premium features not available in my country" → `content_search` or `general_inquiry`
- "I'm on premium but still getting ads" → `premium_billing` (premium feature not working)

---

## Intent 5: download_offline

**Definition:** Problems downloading songs/playlists for offline listening, syncing across devices, or storage issues. The keyword "download", "offline", "sync", or "saved songs" combined with a problem.

**Inclusion criteria:**
- Songs or playlists won't download
- Downloaded songs disappear or get removed
- Offline mode not working
- Sync issues between devices
- Storage limit reached for downloads
- Downloads stuck or very slow

**Exclusion criteria:**
- Streaming (non-downloaded) playback issues → `playback_issue`
- General app crashes → `app_bug`
- Trying to install the Spotify app → `app_bug`

**Representative examples:**
- "i'm able to play the playlist on my computer but it won't sync to my phone"
- "my downloaded songs keep disappearing every time i restart the app"
- "offline mode says i have no songs even though i downloaded them"

**Confusing edge cases:**
- "downloaded songs won't play" → `playback_issue` if the audio is broken; `download_offline` if the downloads themselves are missing/corrupted
- "I'm offline and can't browse" → `download_offline`
- "trying to download the app" → `app_bug` (installing the app, not downloading music)
- "downloads are fine but I can't browse" → `download_offline`

---

## Intent 6: content_search

**Definition:** Can't find a specific song, album, artist, or playlist; content missing from search results; content available in one region but not another; playlist missing or deleted; library not showing recently added content.

**Inclusion criteria:**
- Song, album, or artist not appearing in search
- Content available in one region but not another (licensing/geo)
- Playlist missing or deleted
- Library not showing recently added content
- Collaborative playlist issues
- Podcast or episode not found
- "This song is not available" message for specific content

**Exclusion criteria:**
- Downloaded content not playing → `download_offline`
- Playback issues after finding content → `playback_issue`
- Can't find a UI element or settings menu → `app_bug` or `general_inquiry`

**Representative examples:**
- "there is no way to manage Albums in the App, recently added ones don't show"
- "i can't find this album on spotify but it's on apple music"
- "my playlist disappeared and i can't find it anywhere"

**Confusing edge cases:**
- "I can't find the settings menu" → `app_bug` or `general_inquiry` (UI navigation)
- "song is not available" when playing → `playback_issue` if it was playing before; `content_search` if the content is genuinely missing from the catalogue
- "Spotify isn't available in my country" → `content_search` (service availability)
- "song history is always empty" → `content_search` (library/history feature missing)

---

## Intent 7: general_inquiry

**Definition:** General questions about Spotify features, how-to questions, feedback (positive or negative), questions about device compatibility, or messages that don't describe a specific technical problem. This is the catch-all intent — use it only when no other intent fits.

**Inclusion criteria:**
- How-to questions about app features
- Feature requests or suggestions
- General feedback (positive or negative)
- Questions about availability in a country/region
- Questions about device compatibility
- Follow-up confirmations ("it's working now, thanks!")
- Messages providing device/version info in response to a support question

**Exclusion criteria:**
- Any message describing a specific technical problem → appropriate specific intent
- Billing questions → `premium_billing`
- Messages that genuinely don't fit any category → `other`

**Representative examples:**
- "can you make it a feature that i can enable crossfade on mobile"
- "is spotify available in india yet"
- "love the new update, just wanted to say thanks"

**Confusing edge cases:**
- "thank you, it's fixed now" → `general_inquiry` (resolution confirmation)
- "I don't understand why there's a download limit" → `download_offline` if experiencing the limit as a problem; `general_inquiry` if just asking why
- "iPhone 8, iOS 11.0.3, Spotify version 8.4.22.515" → `general_inquiry` (device info reply, no problem stated)
- Short URL-only messages → `general_inquiry` (cannot determine intent)

---

## Intent 8: other

**Definition:** Messages that do not fit any of the 7 support intents above. Includes: messages in languages other than English where intent cannot be determined, completely off-topic messages, spam, messages addressed to the wrong brand, or messages where the intent is genuinely indeterminate even after reading carefully.

**Inclusion criteria:**
- Message is not about Spotify at all
- Message is addressed to a different brand
- Message is spam or automated
- Message is in a language where intent cannot be determined
- Message is so short or context-dependent that no intent can be assigned (e.g. "yes", "ok", "no")
- Message is a pure URL with no text

**Exclusion criteria:**
- Short follow-up messages where the underlying issue can be inferred → use the inferred intent
- Messages that are vague but clearly about a Spotify problem → use `general_inquiry`

**Representative examples:**
- "yes" (standalone, no context)
- "ok" (standalone, no context)
- A tweet addressed to @AppleSupport that was mistakenly included
- Pure emoji with no text

**Confusing edge cases:**
- "Thanks!" as a standalone message → `general_inquiry` (resolution confirmation, not truly `other`)
- Very short messages like "help" → `general_inquiry` (clearly about needing support)
- "https://t.co/xxxxxx" (URL only) → `other` if no text; `general_inquiry` if context suggests it's a screenshot

---

## Priority order for ambiguous cases

When a message matches multiple intents, use this priority order:

1. `premium_billing` — financial issues take priority (HIGH RISK)
2. `account_login` — account access issues take priority (HIGH RISK)
3. `playback_issue`
4. `app_bug`
5. `download_offline`
6. `content_search`
7. `general_inquiry`
8. `other` — only if nothing else fits

**Exception:** If a message clearly has two distinct problems (e.g. "app crashes AND I was charged twice"), pick the more urgent/primary issue and note the secondary in `labeling_notes`.

---

## Escalation guidance

| Intent | Default routing | Rationale |
|--------|----------------|-----------|
| premium_billing | ESCALATE | Financial harm risk |
| account_login | ESCALATE | Account lockout risk |
| playback_issue | AUTO_HANDLE | Safe troubleshooting steps |
| app_bug | AUTO_HANDLE | Safe reinstall/restart advice |
| download_offline | AUTO_HANDLE | Safe sync/download troubleshooting |
| content_search | AUTO_HANDLE | Informational responses |
| general_inquiry | AUTO_HANDLE | How-to and feedback responses |
| other | ESCALATE | Cannot determine intent safely |

Note: AUTO_HANDLE is subject to confidence and evidence thresholds. Low-confidence or low-evidence cases escalate regardless of intent.
