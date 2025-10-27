# Grey Matter Society Big/Little Program - Form Schema

| Column Name                                         | Data Type      | Answer Choices / Notes                                                                 | Applies To         |
|-----------------------------------------------------|----------------|---------------------------------------------------------------------------------------|--------------------|
| Timestamp                                           | datetime       | Auto-generated                                                                       | All                |
| Full Name                                           | string         | Free text                                                                             | All                |
| Email (School or University Email)                  | string         | Free text (should be valid email, .edu for mentors preferred)                         | All                |
| Gender                                              | string         | Male, Female, Other, Prefer not to say                                                | All                |
| Are you part of the LGBTQ+ community?               | string         | Yes, No, Prefer not to say                                                            | All                |
| Ethnicity                                           | string         | White, Black, Asian, Hispanic, Two or more races, Other, Prefer not to say            | All                |
| State                                               | string         | NJ, CA, IL, ... (US states)                                                           | All                |
| Do you want to be a Mentor or Mentee?               | string         | Mentor, Mentee                                                                       | All                |
| Availability [Monday]                               | list[string]   | Morning, Afternoon, Evening, Night                                                    | All                |
| Availability [Tuesday]                              | list[string]   | Morning, Afternoon, Evening, Night                                                    | All                |
| Availability [Wednesday]                            | list[string]   | Morning, Afternoon, Evening, Night                                                    | All                |
| Availability [Thursday]                             | list[string]   | Morning, Afternoon, Evening, Night                                                    | All                |
| Availability [Friday]                               | list[string]   | Morning, Afternoon, Evening, Night                                                    | All                |
| Availability [Saturday]                             | list[string]   | Morning, Afternoon, Evening, Night                                                    | All                |
| Availability [Sunday]                               | list[string]   | Morning, Afternoon, Evening, Night                                                    | All                |
| MBTI                                                | string         | INTJ, INTP, ENTJ, ENTP, INFJ, INFP, ENFJ, ENFP, ISTJ, ISFJ, ESTJ, ESFJ, ISTP, ISFP, ESTP, ESFP | All      |
| Interests/Hobbies                                  | list[string]   | Free text, comma-separated                                                            | All                |
| Goals                                               | string         | Free text                                                                             | All                |
| Would you like to be matched with someone of similar ethnic demographics? | string | Yes, No, No Preference                                                               | All                |
| Would you like to be matched with someone of similar gender?            | string | Yes, No, No Preference                                                               | All                |
| Would you like to be matched with someone from the LGBTQ+ community?    | string | Yes, No, No Preference                                                               | All                |
| University                                          | string         | Rutgers University, UC San Diego, Northwestern University, ...                        | Mentor only        |
| Class Year                                          | integer        | 2025, 2026, 2027, 2028, 2029                                                          | Mentor only        |
| How many mentees do you want?                       | integer        | 1, 2, 3                                                                               | Mentor only        |
| High School                                         | string         | Piscatway High School, ...                                                            | Mentee only        |

**Notes:**
- Mentor-specific fields (`University`, `Class Year`, `How many mentees do you want?`) are only filled if "Mentor" is selected; otherwise, they are null.
- Mentee-specific field (`High School`) is only filled if "Mentee" is selected; otherwise, it is null.
- Availability fields are multi-select per day.
- All other fields apply to both mentors and mentees.