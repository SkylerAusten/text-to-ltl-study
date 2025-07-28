-- 1. Leaf tables first (most dependent)
DELETE FROM expression_classification_agreement;
DELETE FROM word_reflection;

-- 2. Then classification data
DELETE FROM word_classification;

-- 3. Then expressions and descriptions tied to sessions
DELETE FROM candidate_expression;
DELETE FROM text_description;

-- 4. Then tool sessions
DELETE FROM tool_session;

-- 5. Then user-dependent follow-up responses
DELETE FROM follow_up_response;

-- 6. Finally, delete users
DELETE FROM user;
