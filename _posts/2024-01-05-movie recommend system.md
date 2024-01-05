본 자료는 다음 링크의 내용을 참고하였습니다.

- Reference : https://www.kaggle.com/code/ibtesama/getting-started-with-a-movie-recommendation-system

# 영화 추천 시스템

1. Demographic Filtering (인구통계학적 필터링)
1. Content Based Filtering (컨텐츠 기반 필터링)
1. Collaborative Filtering (협업 필터링)

## 1. Demographic Filtering (인구통계학적 필터링)


```python
import pandas as pd
import numpy as np

df1 = pd.read_csv('tmdb_5000_credits.csv')
df2 = pd.read_csv('tmdb_5000_movies.csv')
```


```python
df1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movie_id</th>
      <th>title</th>
      <th>cast</th>
      <th>crew</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19995</td>
      <td>Avatar</td>
      <td>[{"cast_id": 242, "character": "Jake Sully", "...</td>
      <td>[{"credit_id": "52fe48009251416c750aca23", "de...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>285</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>[{"cast_id": 4, "character": "Captain Jack Spa...</td>
      <td>[{"credit_id": "52fe4232c3a36847f800b579", "de...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>206647</td>
      <td>Spectre</td>
      <td>[{"cast_id": 1, "character": "James Bond", "cr...</td>
      <td>[{"credit_id": "54805967c3a36829b5002c41", "de...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>49026</td>
      <td>The Dark Knight Rises</td>
      <td>[{"cast_id": 2, "character": "Bruce Wayne / Ba...</td>
      <td>[{"credit_id": "52fe4781c3a36847f81398c3", "de...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>49529</td>
      <td>John Carter</td>
      <td>[{"cast_id": 5, "character": "John Carter", "c...</td>
      <td>[{"credit_id": "52fe479ac3a36847f813eaa3", "de...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>keywords</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>popularity</th>
      <th>production_companies</th>
      <th>production_countries</th>
      <th>release_date</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>237000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://www.avatarmovie.com/</td>
      <td>19995</td>
      <td>[{"id": 1463, "name": "culture clash"}, {"id":...</td>
      <td>en</td>
      <td>Avatar</td>
      <td>In the 22nd century, a paraplegic Marine is di...</td>
      <td>150.437577</td>
      <td>[{"name": "Ingenious Film Partners", "id": 289...</td>
      <td>[{"iso_3166_1": "US", "name": "United States o...</td>
      <td>2009-12-10</td>
      <td>2787965087</td>
      <td>162.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}, {"iso...</td>
      <td>Released</td>
      <td>Enter the World of Pandora.</td>
      <td>Avatar</td>
      <td>7.2</td>
      <td>11800</td>
    </tr>
    <tr>
      <th>1</th>
      <td>300000000</td>
      <td>[{"id": 12, "name": "Adventure"}, {"id": 14, "...</td>
      <td>http://disney.go.com/disneypictures/pirates/</td>
      <td>285</td>
      <td>[{"id": 270, "name": "ocean"}, {"id": 726, "na...</td>
      <td>en</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>Captain Barbossa, long believed to be dead, ha...</td>
      <td>139.082615</td>
      <td>[{"name": "Walt Disney Pictures", "id": 2}, {"...</td>
      <td>[{"iso_3166_1": "US", "name": "United States o...</td>
      <td>2007-05-19</td>
      <td>961000000</td>
      <td>169.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>At the end of the world, the adventure begins.</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>6.9</td>
      <td>4500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>245000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://www.sonypictures.com/movies/spectre/</td>
      <td>206647</td>
      <td>[{"id": 470, "name": "spy"}, {"id": 818, "name...</td>
      <td>en</td>
      <td>Spectre</td>
      <td>A cryptic message from Bond’s past sends him o...</td>
      <td>107.376788</td>
      <td>[{"name": "Columbia Pictures", "id": 5}, {"nam...</td>
      <td>[{"iso_3166_1": "GB", "name": "United Kingdom"...</td>
      <td>2015-10-26</td>
      <td>880674609</td>
      <td>148.0</td>
      <td>[{"iso_639_1": "fr", "name": "Fran\u00e7ais"},...</td>
      <td>Released</td>
      <td>A Plan No One Escapes</td>
      <td>Spectre</td>
      <td>6.3</td>
      <td>4466</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1.shape, df2.shape #테이터의 수가 같은지 다른지를 확인 합니다.
```




    ((4803, 4), (4803, 20))




```python
df1['title'].equals(df2['title']) #두 데이터의 '타이틀'이 같은지 확인 합니다.
```




    True




```python
df1.columns
```




    Index(['movie_id', 'title', 'cast', 'crew'], dtype='object')




```python
df1.columns = ['id', 'title', 'cast', 'crew'] #'movie_id'를 'id'로 변경 두데이터의 컬럼 제목을 통일 시켜 줍니다.
df1.columns
```




    Index(['id', 'title', 'cast', 'crew'], dtype='object')




```python
df1[['id', 'cast', 'crew']]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>cast</th>
      <th>crew</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19995</td>
      <td>[{"cast_id": 242, "character": "Jake Sully", "...</td>
      <td>[{"credit_id": "52fe48009251416c750aca23", "de...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>285</td>
      <td>[{"cast_id": 4, "character": "Captain Jack Spa...</td>
      <td>[{"credit_id": "52fe4232c3a36847f800b579", "de...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>206647</td>
      <td>[{"cast_id": 1, "character": "James Bond", "cr...</td>
      <td>[{"credit_id": "54805967c3a36829b5002c41", "de...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>49026</td>
      <td>[{"cast_id": 2, "character": "Bruce Wayne / Ba...</td>
      <td>[{"credit_id": "52fe4781c3a36847f81398c3", "de...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>49529</td>
      <td>[{"cast_id": 5, "character": "John Carter", "c...</td>
      <td>[{"credit_id": "52fe479ac3a36847f813eaa3", "de...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4798</th>
      <td>9367</td>
      <td>[{"cast_id": 1, "character": "El Mariachi", "c...</td>
      <td>[{"credit_id": "52fe44eec3a36847f80b280b", "de...</td>
    </tr>
    <tr>
      <th>4799</th>
      <td>72766</td>
      <td>[{"cast_id": 1, "character": "Buzzy", "credit_...</td>
      <td>[{"credit_id": "52fe487dc3a368484e0fb013", "de...</td>
    </tr>
    <tr>
      <th>4800</th>
      <td>231617</td>
      <td>[{"cast_id": 8, "character": "Oliver O\u2019To...</td>
      <td>[{"credit_id": "52fe4df3c3a36847f8275ecf", "de...</td>
    </tr>
    <tr>
      <th>4801</th>
      <td>126186</td>
      <td>[{"cast_id": 3, "character": "Sam", "credit_id...</td>
      <td>[{"credit_id": "52fe4ad9c3a368484e16a36b", "de...</td>
    </tr>
    <tr>
      <th>4802</th>
      <td>25975</td>
      <td>[{"cast_id": 3, "character": "Herself", "credi...</td>
      <td>[{"credit_id": "58ce021b9251415a390165d9", "de...</td>
    </tr>
  </tbody>
</table>
<p>4803 rows × 3 columns</p>
</div>



df1데이터의 'id','cast','crew'를 df2데이터의 id를 기준으로 끝 열에 붙이기


```python
df2 = df2.merge(df1[['id', 'cast', 'crew']], on='id') #'id'를 기준으로 df2를 붙이기
df2.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>keywords</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>popularity</th>
      <th>production_companies</th>
      <th>...</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>cast</th>
      <th>crew</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>237000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://www.avatarmovie.com/</td>
      <td>19995</td>
      <td>[{"id": 1463, "name": "culture clash"}, {"id":...</td>
      <td>en</td>
      <td>Avatar</td>
      <td>In the 22nd century, a paraplegic Marine is di...</td>
      <td>150.437577</td>
      <td>[{"name": "Ingenious Film Partners", "id": 289...</td>
      <td>...</td>
      <td>2787965087</td>
      <td>162.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}, {"iso...</td>
      <td>Released</td>
      <td>Enter the World of Pandora.</td>
      <td>Avatar</td>
      <td>7.2</td>
      <td>11800</td>
      <td>[{"cast_id": 242, "character": "Jake Sully", "...</td>
      <td>[{"credit_id": "52fe48009251416c750aca23", "de...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>300000000</td>
      <td>[{"id": 12, "name": "Adventure"}, {"id": 14, "...</td>
      <td>http://disney.go.com/disneypictures/pirates/</td>
      <td>285</td>
      <td>[{"id": 270, "name": "ocean"}, {"id": 726, "na...</td>
      <td>en</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>Captain Barbossa, long believed to be dead, ha...</td>
      <td>139.082615</td>
      <td>[{"name": "Walt Disney Pictures", "id": 2}, {"...</td>
      <td>...</td>
      <td>961000000</td>
      <td>169.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>At the end of the world, the adventure begins.</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>6.9</td>
      <td>4500</td>
      <td>[{"cast_id": 4, "character": "Captain Jack Spa...</td>
      <td>[{"credit_id": "52fe4232c3a36847f800b579", "de...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>245000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://www.sonypictures.com/movies/spectre/</td>
      <td>206647</td>
      <td>[{"id": 470, "name": "spy"}, {"id": 818, "name...</td>
      <td>en</td>
      <td>Spectre</td>
      <td>A cryptic message from Bond’s past sends him o...</td>
      <td>107.376788</td>
      <td>[{"name": "Columbia Pictures", "id": 5}, {"nam...</td>
      <td>...</td>
      <td>880674609</td>
      <td>148.0</td>
      <td>[{"iso_639_1": "fr", "name": "Fran\u00e7ais"},...</td>
      <td>Released</td>
      <td>A Plan No One Escapes</td>
      <td>Spectre</td>
      <td>6.3</td>
      <td>4466</td>
      <td>[{"cast_id": 1, "character": "James Bond", "cr...</td>
      <td>[{"credit_id": "54805967c3a36829b5002c41", "de...</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 22 columns</p>
</div>



영화 1 : 영화의 평점이 10/10 -> 5명이 평가

영화 2 : 영화의 평점이 8/10 -> 500명이 평가

![image.png](attachment:image.png)

#### C: 모든 영화에 대한 평점 평균


```python
C = df2['vote_average'].mean()
C
```




    6.092171559442011



#### m: 차트에 나열되는 데 필요한 최소 투표 수  
####      (평가하기에 너무 적은 득표를 차지하는 영화는 제외해야 한다)


```python
m = df2['vote_count'].quantile(0.9) #상위 10% 영화 만 뽑아준다.
m
```




    1838.4000000000015



#### 평점 갯수가 1838개 이상인것만 가져오기


```python
q_movies = df2.copy().loc[df2['vote_count'] >= m]
q_movies.shape
```




    (481, 22)




```python
q_movies['vote_count'].sort_values()
```




    2585     1840
    195      1851
    2454     1859
    597      1862
    1405     1864
            ...  
    788     10995
    16      11776
    0       11800
    65      12002
    96      13752
    Name: vote_count, Length: 481, dtype: int64



#### 가중평가 함수 작성하기


```python
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)
```


```python
q_movies['score'] = q_movies.apply(weighted_rating, axis=1) # axis=1은 row단위로 데이터를 만들어 준다
q_movies.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>keywords</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>popularity</th>
      <th>production_companies</th>
      <th>...</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>cast</th>
      <th>crew</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>237000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://www.avatarmovie.com/</td>
      <td>19995</td>
      <td>[{"id": 1463, "name": "culture clash"}, {"id":...</td>
      <td>en</td>
      <td>Avatar</td>
      <td>In the 22nd century, a paraplegic Marine is di...</td>
      <td>150.437577</td>
      <td>[{"name": "Ingenious Film Partners", "id": 289...</td>
      <td>...</td>
      <td>162.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}, {"iso...</td>
      <td>Released</td>
      <td>Enter the World of Pandora.</td>
      <td>Avatar</td>
      <td>7.2</td>
      <td>11800</td>
      <td>[{"cast_id": 242, "character": "Jake Sully", "...</td>
      <td>[{"credit_id": "52fe48009251416c750aca23", "de...</td>
      <td>7.050669</td>
    </tr>
    <tr>
      <th>1</th>
      <td>300000000</td>
      <td>[{"id": 12, "name": "Adventure"}, {"id": 14, "...</td>
      <td>http://disney.go.com/disneypictures/pirates/</td>
      <td>285</td>
      <td>[{"id": 270, "name": "ocean"}, {"id": 726, "na...</td>
      <td>en</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>Captain Barbossa, long believed to be dead, ha...</td>
      <td>139.082615</td>
      <td>[{"name": "Walt Disney Pictures", "id": 2}, {"...</td>
      <td>...</td>
      <td>169.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>At the end of the world, the adventure begins.</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>6.9</td>
      <td>4500</td>
      <td>[{"cast_id": 4, "character": "Captain Jack Spa...</td>
      <td>[{"credit_id": "52fe4232c3a36847f800b579", "de...</td>
      <td>6.665696</td>
    </tr>
    <tr>
      <th>2</th>
      <td>245000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://www.sonypictures.com/movies/spectre/</td>
      <td>206647</td>
      <td>[{"id": 470, "name": "spy"}, {"id": 818, "name...</td>
      <td>en</td>
      <td>Spectre</td>
      <td>A cryptic message from Bond’s past sends him o...</td>
      <td>107.376788</td>
      <td>[{"name": "Columbia Pictures", "id": 5}, {"nam...</td>
      <td>...</td>
      <td>148.0</td>
      <td>[{"iso_639_1": "fr", "name": "Fran\u00e7ais"},...</td>
      <td>Released</td>
      <td>A Plan No One Escapes</td>
      <td>Spectre</td>
      <td>6.3</td>
      <td>4466</td>
      <td>[{"cast_id": 1, "character": "James Bond", "cr...</td>
      <td>[{"credit_id": "54805967c3a36829b5002c41", "de...</td>
      <td>6.239396</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 23 columns</p>
</div>




```python
q_movies = q_movies.sort_values('score', ascending=False) # ascending은 내림차순 정렬
q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10) #Top10영화만 가져오기
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1881</th>
      <td>The Shawshank Redemption</td>
      <td>8205</td>
      <td>8.5</td>
      <td>8.059258</td>
    </tr>
    <tr>
      <th>662</th>
      <td>Fight Club</td>
      <td>9413</td>
      <td>8.3</td>
      <td>7.939256</td>
    </tr>
    <tr>
      <th>65</th>
      <td>The Dark Knight</td>
      <td>12002</td>
      <td>8.2</td>
      <td>7.920020</td>
    </tr>
    <tr>
      <th>3232</th>
      <td>Pulp Fiction</td>
      <td>8428</td>
      <td>8.3</td>
      <td>7.904645</td>
    </tr>
    <tr>
      <th>96</th>
      <td>Inception</td>
      <td>13752</td>
      <td>8.1</td>
      <td>7.863239</td>
    </tr>
    <tr>
      <th>3337</th>
      <td>The Godfather</td>
      <td>5893</td>
      <td>8.4</td>
      <td>7.851236</td>
    </tr>
    <tr>
      <th>95</th>
      <td>Interstellar</td>
      <td>10867</td>
      <td>8.1</td>
      <td>7.809479</td>
    </tr>
    <tr>
      <th>809</th>
      <td>Forrest Gump</td>
      <td>7927</td>
      <td>8.2</td>
      <td>7.803188</td>
    </tr>
    <tr>
      <th>329</th>
      <td>The Lord of the Rings: The Return of the King</td>
      <td>8064</td>
      <td>8.1</td>
      <td>7.727243</td>
    </tr>
    <tr>
      <th>1990</th>
      <td>The Empire Strikes Back</td>
      <td>5879</td>
      <td>8.2</td>
      <td>7.697884</td>
    </tr>
  </tbody>
</table>
</div>



#### 데이터 시각화 해보기(Top 10 영화)

인기도 선정 방식 https://developer.themoviedb.org/docs/popularity-and-trending


```python
pop= df2.sort_values('popularity', ascending=False)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))

plt.barh(pop['title'].head(10),pop['popularity'].head(10), align='center',
        color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")
```




    Text(0.5, 1.0, 'Popular Movies')




    
![popular movies](/images/output_29_1.png)
    


## 2. Content Based Filtering (컨텐츠 기반 필터링)

### 줄거리 기반 추천


```python
df2['overview'].head(5)
```




    0    In the 22nd century, a paraplegic Marine is di...
    1    Captain Barbossa, long believed to be dead, ha...
    2    A cryptic message from Bond’s past sends him o...
    3    Following the death of District Attorney Harve...
    4    John Carter is a war-weary, former military ca...
    Name: overview, dtype: object



#### NLP

Bag Of Words - BOW

문장1 : I am a boy

문장2 : I am a girl

I(2), am(2), a(2), boy(1), girl(1)

        
        
        I    am   a   boy    girl
 문장1  1    1    1    1      0   (1,1,1,1,0)
 (I am a boy)
 
 문장2  1    1    1    0      1   (1,1,1,0,1)
 (I am a girl)
 
 피처 벡터화.
 
 
 문서 100개
 모든 문서에서 나온 단어 10,000 개
 100 * 10,000 = 100만
 
        단어1, 단어2, 단어3, 단어4, .... 단어 10000
문서1    1       1       3    0    
문서2
문서3
..
문서100

1. TfidfVectorizer (TF-IDF 기반의 벡터화)
2. CountVectorizer (모든 단어에 대해서 있는 그대로 갯수를 세는것)


TF-IDF 벡터 import


```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
```


```python
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
ENGLISH_STOP_WORDS
```




    frozenset({'a',
               'about',
               'above',
               'across',
               'after',
               'afterwards',
               'again',
               'against',
               'all',
               'almost',
               'alone',
               'along',
               'already',
               'also',
               'although',
               'always',
               'am',
               'among',
               'amongst',
               'amoungst',
               'amount',
               'an',
               'and',
               'another',
               'any',
               'anyhow',
               'anyone',
               'anything',
               'anyway',
               'anywhere',
               'are',
               'around',
               'as',
               'at',
               'back',
               'be',
               'became',
               'because',
               'become',
               'becomes',
               'becoming',
               'been',
               'before',
               'beforehand',
               'behind',
               'being',
               'below',
               'beside',
               'besides',
               'between',
               'beyond',
               'bill',
               'both',
               'bottom',
               'but',
               'by',
               'call',
               'can',
               'cannot',
               'cant',
               'co',
               'con',
               'could',
               'couldnt',
               'cry',
               'de',
               'describe',
               'detail',
               'do',
               'done',
               'down',
               'due',
               'during',
               'each',
               'eg',
               'eight',
               'either',
               'eleven',
               'else',
               'elsewhere',
               'empty',
               'enough',
               'etc',
               'even',
               'ever',
               'every',
               'everyone',
               'everything',
               'everywhere',
               'except',
               'few',
               'fifteen',
               'fifty',
               'fill',
               'find',
               'fire',
               'first',
               'five',
               'for',
               'former',
               'formerly',
               'forty',
               'found',
               'four',
               'from',
               'front',
               'full',
               'further',
               'get',
               'give',
               'go',
               'had',
               'has',
               'hasnt',
               'have',
               'he',
               'hence',
               'her',
               'here',
               'hereafter',
               'hereby',
               'herein',
               'hereupon',
               'hers',
               'herself',
               'him',
               'himself',
               'his',
               'how',
               'however',
               'hundred',
               'i',
               'ie',
               'if',
               'in',
               'inc',
               'indeed',
               'interest',
               'into',
               'is',
               'it',
               'its',
               'itself',
               'keep',
               'last',
               'latter',
               'latterly',
               'least',
               'less',
               'ltd',
               'made',
               'many',
               'may',
               'me',
               'meanwhile',
               'might',
               'mill',
               'mine',
               'more',
               'moreover',
               'most',
               'mostly',
               'move',
               'much',
               'must',
               'my',
               'myself',
               'name',
               'namely',
               'neither',
               'never',
               'nevertheless',
               'next',
               'nine',
               'no',
               'nobody',
               'none',
               'noone',
               'nor',
               'not',
               'nothing',
               'now',
               'nowhere',
               'of',
               'off',
               'often',
               'on',
               'once',
               'one',
               'only',
               'onto',
               'or',
               'other',
               'others',
               'otherwise',
               'our',
               'ours',
               'ourselves',
               'out',
               'over',
               'own',
               'part',
               'per',
               'perhaps',
               'please',
               'put',
               'rather',
               're',
               'same',
               'see',
               'seem',
               'seemed',
               'seeming',
               'seems',
               'serious',
               'several',
               'she',
               'should',
               'show',
               'side',
               'since',
               'sincere',
               'six',
               'sixty',
               'so',
               'some',
               'somehow',
               'someone',
               'something',
               'sometime',
               'sometimes',
               'somewhere',
               'still',
               'such',
               'system',
               'take',
               'ten',
               'than',
               'that',
               'the',
               'their',
               'them',
               'themselves',
               'then',
               'thence',
               'there',
               'thereafter',
               'thereby',
               'therefore',
               'therein',
               'thereupon',
               'these',
               'they',
               'thick',
               'thin',
               'third',
               'this',
               'those',
               'though',
               'three',
               'through',
               'throughout',
               'thru',
               'thus',
               'to',
               'together',
               'too',
               'top',
               'toward',
               'towards',
               'twelve',
               'twenty',
               'two',
               'un',
               'under',
               'until',
               'up',
               'upon',
               'us',
               'very',
               'via',
               'was',
               'we',
               'well',
               'were',
               'what',
               'whatever',
               'when',
               'whence',
               'whenever',
               'where',
               'whereafter',
               'whereas',
               'whereby',
               'wherein',
               'whereupon',
               'wherever',
               'whether',
               'which',
               'while',
               'whither',
               'who',
               'whoever',
               'whole',
               'whom',
               'whose',
               'why',
               'will',
               'with',
               'within',
               'without',
               'would',
               'yet',
               'you',
               'your',
               'yours',
               'yourself',
               'yourselves'})




```python
df2['overview'].isnull().values.any() #null데이터를 검출
```




    True




```python
df2['overview'] = df2['overview'].fillna('') #자동으로 null값을 찾아서 빈데이터롤 넣어 주기
```


```python
tfidf_matrix = tfidf.fit_transform(df2['overview'])
tfidf_matrix.shape
```




    (4803, 20978)




```python
tfidf_matrix
```




    <4803x20978 sparse matrix of type '<class 'numpy.float64'>'
    	with 125840 stored elements in Compressed Sparse Row format>



코사인 유사도(단어 유사도 확인)


```python
from sklearn.metrics.pairwise import linear_kernel

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim
```




    array([[1.        , 0.        , 0.        , ..., 0.        , 0.        ,
            0.        ],
           [0.        , 1.        , 0.        , ..., 0.02160533, 0.        ,
            0.        ],
           [0.        , 0.        , 1.        , ..., 0.01488159, 0.        ,
            0.        ],
           ...,
           [0.        , 0.02160533, 0.01488159, ..., 1.        , 0.01609091,
            0.00701914],
           [0.        , 0.        , 0.        , ..., 0.01609091, 1.        ,
            0.01171696],
           [0.        , 0.        , 0.        , ..., 0.00701914, 0.01171696,
            1.        ]])



| | 문장1 | 문장2 | 문장3 |
|---|---|---|---|
|문장1|1|0.3|0.8|
|문장2|0.3|1|0.5|
|문장3|0.8|0.5|1|


```python
cosine_sim.shape
```




    (4803, 4803)



Series 만들기(영화 제목을 가지고 데이터에서 몇번째에 해당하는지 알기 위한 데이터 만들기)


```python
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates() #drop_duplicates 중복데이터 삭제
indices                                                              #영화 제목에 인덱스를 부여함
```




    title
    Avatar                                         0
    Pirates of the Caribbean: At World's End       1
    Spectre                                        2
    The Dark Knight Rises                          3
    John Carter                                    4
                                                ... 
    El Mariachi                                 4798
    Newlyweds                                   4799
    Signed, Sealed, Delivered                   4800
    Shanghai Calling                            4801
    My Date with Drew                           4802
    Length: 4803, dtype: int64




```python
indices['Avatar'] #영화제목을 입력받아 index를 출력
indices['The Dark Knight Rises']
```




    3




```python
df2.iloc[[3]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>keywords</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>popularity</th>
      <th>production_companies</th>
      <th>...</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>cast</th>
      <th>crew</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>250000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 80, "nam...</td>
      <td>http://www.thedarkknightrises.com/</td>
      <td>49026</td>
      <td>[{"id": 849, "name": "dc comics"}, {"id": 853,...</td>
      <td>en</td>
      <td>The Dark Knight Rises</td>
      <td>Following the death of District Attorney Harve...</td>
      <td>112.31295</td>
      <td>[{"name": "Legendary Pictures", "id": 923}, {"...</td>
      <td>...</td>
      <td>1084939099</td>
      <td>165.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>The Legend Ends</td>
      <td>The Dark Knight Rises</td>
      <td>7.6</td>
      <td>9106</td>
      <td>[{"cast_id": 2, "character": "Bruce Wayne / Ba...</td>
      <td>[{"credit_id": "52fe4781c3a36847f81398c3", "de...</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 22 columns</p>
</div>




```python
# 영화의 제목을 입력받으면 코사인 유사도를 통해서 가장 유사도가 높은 상위 10개의 영화 목록 반환
def get_recommendations(title, cosine_sim=cosine_sim):
    # 영화 제목을 통해서 전체 데이터 기준 그 영화의 index 값을 얻기
    idx = indices[title]
    
    # 코사인 유사도 매트릭스 (cosine_sim) 에서 idx 에 해당하는 데이터를 (idx, 유사도) 형태로 얻기
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # 코사인 유사도 기준으로 내림차순 정렬
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # 자기 자신을 제외한 10개의 추천 영화를 슬라이싱
    sim_scores = sim_scores[1:11]
    
    # 추천 영화 목록 10개의 인덱스 정보 추출
    movie_indices = [i[0] for i in sim_scores]
    
    # 인덱스 정보를 통해 영화 제목 추출
    return df2['title'].iloc[movie_indices]
```

함수 내부 코드 설명


```python
test_idx = indices['The Dark Knight Rises'] # 영화 제목을 통해서 전체 데이터 기준 그 영화의 index 값을 얻기
test_idx
```




    3




```python
cosine_sim[3] #인덱스3에 해당하는 코사인 유사도 불러 오기
```




    array([0.02499512, 0.        , 0.        , ..., 0.03386366, 0.04275232,
           0.02269198])




```python
test_sim_scores = list(enumerate(cosine_sim[3])) # 코사인 유사도 매트릭스 (cosine_sim) 에서 idx 에 해당하는 데이터를 (idx, 유사도) 형태로 얻기
test_sim_scores
```




    [(0, 0.0249951158376727),
     (1, 0.0),
     (2, 0.0),
     (3, 1.0),
     (4, 0.010433403719159354),
     (5, 0.0051446018158107934),
     (6, 0.01260063243546246),
     (7, 0.026954270578912674),
     (8, 0.02065221688538951),
     (9, 0.1337400906655523),
     (10, 0.0),
     (11, 0.0),
     (12, 0.0),
     (13, 0.0),
     (14, 0.0),
     (15, 0.004071333922512107),
     (16, 0.021121093874993183),
     (17, 0.0),
     (18, 0.006768893195007471),
     (19, 0.010765175685064708),
     (20, 0.007178266390761152),
     (21, 0.033380775071488206),
     (22, 0.0),
     (23, 0.0),
     (24, 0.019238168304196293),
     (25, 0.017013388161368177),
     (26, 0.018845673291717265),
     (27, 0.0),
     (28, 0.008351231142809445),
     (29, 0.02582248131068942),
     (30, 0.017549884233472455),
     (31, 0.029947529512076927),
     (32, 0.021708602796233892),
     (33, 0.0),
     (34, 0.0),
     (35, 0.0),
     (36, 0.028418448812136628),
     (37, 0.0),
     (38, 0.030631547964966054),
     (39, 0.01088522587631032),
     (40, 0.0),
     (41, 0.021109848007927),
     (42, 0.008087287409335878),
     (43, 0.0),
     (44, 0.0),
     (45, 0.008667206203132952),
     (46, 0.0),
     (47, 0.008370116577193958),
     (48, 0.0),
     (49, 0.0),
     (50, 0.028358356657578104),
     (51, 0.0044195546647069804),
     (52, 0.016300101955129633),
     (53, 0.014997203964410096),
     (54, 0.0),
     (55, 0.0),
     (56, 0.04956374277303492),
     (57, 0.011455415588688133),
     (58, 0.026549302574340555),
     (59, 0.0),
     (60, 0.008984081816033438),
     (61, 0.0),
     (62, 0.0),
     (63, 0.0),
     (64, 0.0),
     (65, 0.30151176591665485),
     (66, 0.005466012755015322),
     (67, 0.0),
     (68, 0.0),
     (69, 0.0),
     (70, 0.0),
     (71, 0.0),
     (72, 0.0),
     (73, 0.0),
     (74, 0.0),
     (75, 0.0),
     (76, 0.01151474877665648),
     (77, 0.018362187347835526),
     (78, 0.0),
     (79, 0.0046399517384596665),
     (80, 0.02040398688574236),
     (81, 0.0),
     (82, 0.0),
     (83, 0.0),
     (84, 0.011095621330455193),
     (85, 0.03972278831039288),
     (86, 0.02383463768528818),
     (87, 0.0),
     (88, 0.0),
     (89, 0.006901163451631844),
     (90, 0.0),
     (91, 0.03267349033106477),
     (92, 0.013909786991724979),
     (93, 0.024108412591485766),
     (94, 0.019899105691655185),
     (95, 0.0),
     (96, 0.0),
     (97, 0.025191659761195462),
     (98, 0.0),
     (99, 0.0),
     (100, 0.0),
     (101, 0.006008620221617743),
     (102, 0.017293543538943367),
     (103, 0.014765415310900133),
     (104, 0.0),
     (105, 0.006827526424670165),
     (106, 0.0),
     (107, 0.0),
     (108, 0.020837965840053214),
     (109, 0.0),
     (110, 0.0),
     (111, 0.0),
     (112, 0.0),
     (113, 0.01539391139573925),
     (114, 0.0),
     (115, 0.004582195354771824),
     (116, 0.02942847555462785),
     (117, 0.0),
     (118, 0.0),
     (119, 0.16799626199850706),
     (120, 0.004967649962522657),
     (121, 0.01783534576376142),
     (122, 0.01568614172147101),
     (123, 0.027652990258597784),
     (124, 0.0),
     (125, 0.004601262695517274),
     (126, 0.017812053858174758),
     (127, 0.014869787303643058),
     (128, 0.011605892485339277),
     (129, 0.0),
     (130, 0.0280204985055396),
     (131, 0.0),
     (132, 0.011065964977875105),
     (133, 0.0),
     (134, 0.0),
     (135, 0.0),
     (136, 0.05382976624450389),
     (137, 0.008883897054844552),
     (138, 0.0),
     (139, 0.04537230110999062),
     (140, 0.0),
     (141, 0.0),
     (142, 0.0),
     (143, 0.004606862711209118),
     (144, 0.0),
     (145, 0.011124596921338135),
     (146, 0.008317327524856892),
     (147, 0.015811113048024572),
     (148, 0.01930380580447377),
     (149, 0.0),
     (150, 0.005151904214092396),
     (151, 0.0),
     (152, 0.0),
     (153, 0.026348116565001682),
     (154, 0.04108719099965726),
     (155, 0.0),
     (156, 0.0),
     (157, 0.01848918875479597),
     (158, 0.0),
     (159, 0.0),
     (160, 0.056387288475695505),
     (161, 0.0),
     (162, 0.0),
     (163, 0.0),
     (164, 0.0),
     (165, 0.0),
     (166, 0.05328858662752758),
     (167, 0.0),
     (168, 0.028588812451723972),
     (169, 0.023998866252772623),
     (170, 0.04815758597763417),
     (171, 0.017388180483771287),
     (172, 0.04140039692840094),
     (173, 0.017304568426605434),
     (174, 0.005605318477839426),
     (175, 0.0),
     (176, 0.0),
     (177, 0.0),
     (178, 0.010421139129967987),
     (179, 0.03843748677355906),
     (180, 0.018217619313564887),
     (181, 0.0),
     (182, 0.028276895060021155),
     (183, 0.0),
     (184, 0.004017150235440391),
     (185, 0.006705822747118495),
     (186, 0.0),
     (187, 0.0),
     (188, 0.005432026720538357),
     (189, 0.051986848187990306),
     (190, 0.0),
     (191, 0.0),
     (192, 0.03244984001292098),
     (193, 0.00785286457851951),
     (194, 0.0),
     (195, 0.005824255269702452),
     (196, 0.0076412385986681965),
     (197, 0.0),
     (198, 0.05321038424109084),
     (199, 0.0),
     (200, 0.0),
     (201, 0.0),
     (202, 0.0),
     (203, 0.0),
     (204, 0.0),
     (205, 0.007258719582486187),
     (206, 0.0),
     (207, 0.0),
     (208, 0.0),
     (209, 0.023727382552120816),
     (210, 0.13045537014449818),
     (211, 0.0),
     (212, 0.03732491379402199),
     (213, 0.0),
     (214, 0.0),
     (215, 0.037223554058902084),
     (216, 0.0),
     (217, 0.015691196579673106),
     (218, 0.0),
     (219, 0.0),
     (220, 0.0),
     (221, 0.0),
     (222, 0.0),
     (223, 0.007909063391451093),
     (224, 0.02199123210296277),
     (225, 0.023388061689382268),
     (226, 0.0),
     (227, 0.0),
     (228, 0.0),
     (229, 0.015668417779476064),
     (230, 0.022384865249046065),
     (231, 0.0),
     (232, 0.0),
     (233, 0.0),
     (234, 0.005416120563002782),
     (235, 0.0),
     (236, 0.03211950994841882),
     (237, 0.0),
     (238, 0.055637188916219035),
     (239, 0.0),
     (240, 0.0),
     (241, 0.004843532442353235),
     (242, 0.028496309153597243),
     (243, 0.0),
     (244, 0.0),
     (245, 0.0),
     (246, 0.037764821851882846),
     (247, 0.0),
     (248, 0.0067984202343237534),
     (249, 0.0),
     (250, 0.0),
     (251, 0.0),
     (252, 0.012633399991584724),
     (253, 0.0035604418677455583),
     (254, 0.0),
     (255, 0.010122999001271669),
     (256, 0.013403248925391863),
     (257, 0.0),
     (258, 0.002637514549434934),
     (259, 0.0066049471959311245),
     (260, 0.0),
     (261, 0.0),
     (262, 0.03981572576257154),
     (263, 0.0),
     (264, 0.008799376722262713),
     (265, 0.0),
     (266, 0.011482113684829372),
     (267, 0.0),
     (268, 0.0),
     (269, 0.0),
     (270, 0.0),
     (271, 0.0),
     (272, 0.02638073666465946),
     (273, 0.0),
     (274, 0.013794380725284881),
     (275, 0.04699106659570477),
     (276, 0.017347942632579733),
     (277, 0.013927075303072751),
     (278, 0.0),
     (279, 0.02306946675005357),
     (280, 0.014835017084107845),
     (281, 0.04850690747212835),
     (282, 0.011076592401190929),
     (283, 0.02945430864867394),
     (284, 0.0056132959217499515),
     (285, 0.0143752204449751),
     (286, 0.06866383939743481),
     (287, 0.0),
     (288, 0.018954785191161307),
     (289, 0.0030400430800127183),
     (290, 0.026749517782380123),
     (291, 0.0),
     (292, 0.0),
     (293, 0.0),
     (294, 0.0),
     (295, 0.010872296157133941),
     (296, 0.040114221787403814),
     (297, 0.0),
     (298, 0.006069530957620157),
     (299, 0.29857045255396825),
     (300, 0.0),
     (301, 0.0),
     (302, 0.004199020500120462),
     (303, 0.005028626630771381),
     (304, 0.043301851324804685),
     (305, 0.0),
     (306, 0.0),
     (307, 0.031573558104568175),
     (308, 0.0),
     (309, 0.0),
     (310, 0.0033919366519925298),
     (311, 0.010398151316103811),
     (312, 0.0),
     (313, 0.0),
     (314, 0.0),
     (315, 0.004422195303905182),
     (316, 0.011024342093847618),
     (317, 0.0),
     (318, 0.0),
     (319, 0.0),
     (320, 0.0),
     (321, 0.03408187954598553),
     (322, 0.0),
     (323, 0.0),
     (324, 0.034747394754250746),
     (325, 0.006948743879506472),
     (326, 0.0),
     (327, 0.0),
     (328, 0.0),
     (329, 0.01635763790225705),
     (330, 0.0),
     (331, 0.0),
     (332, 0.0),
     (333, 0.0),
     (334, 0.004836941913083933),
     (335, 0.0),
     (336, 0.0),
     (337, 0.0),
     (338, 0.0),
     (339, 0.013704587734984675),
     (340, 0.0),
     (341, 0.005608956462969991),
     (342, 0.03925347760202698),
     (343, 0.0),
     (344, 0.0),
     (345, 0.0),
     (346, 0.0),
     (347, 0.0),
     (348, 0.005554915622312921),
     (349, 0.0),
     (350, 0.0),
     (351, 0.024998796675161295),
     (352, 0.0),
     (353, 0.0),
     (354, 0.006058817421523585),
     (355, 0.024714784595197162),
     (356, 0.007619002081246003),
     (357, 0.013972190936862885),
     (358, 0.0),
     (359, 0.039038636330984644),
     (360, 0.0),
     (361, 0.010722257362842224),
     (362, 0.0076241902807024494),
     (363, 0.0),
     (364, 0.0),
     (365, 0.0),
     (366, 0.0),
     (367, 0.0),
     (368, 0.0),
     (369, 0.03968920576233667),
     (370, 0.02966843532390714),
     (371, 0.0),
     (372, 0.0),
     (373, 0.0),
     (374, 0.0),
     (375, 0.003873755160807044),
     (376, 0.02854165335128062),
     (377, 0.00404045615091612),
     (378, 0.0),
     (379, 0.004117591373836143),
     (380, 0.0),
     (381, 0.0),
     (382, 0.0),
     (383, 0.0),
     (384, 0.0),
     (385, 0.0),
     (386, 0.0),
     (387, 0.0),
     (388, 0.0),
     (389, 0.03743705301718331),
     (390, 0.0),
     (391, 0.0036970647958324433),
     (392, 0.04173737309688232),
     (393, 0.004561138780984189),
     (394, 0.0),
     (395, 0.0),
     (396, 0.0),
     (397, 0.01451115325528235),
     (398, 0.01674976857117613),
     (399, 0.0),
     (400, 0.028068887228114868),
     (401, 0.025352330605289516),
     (402, 0.008515312510534539),
     (403, 0.005380258008061251),
     (404, 0.011527860171776575),
     (405, 0.022747389486920688),
     (406, 0.0),
     (407, 0.0078485245098086),
     (408, 0.0),
     (409, 0.0),
     (410, 0.0),
     (411, 0.0),
     (412, 0.0),
     (413, 0.0),
     (414, 0.009394503853080246),
     (415, 0.022897180758222554),
     (416, 0.0),
     (417, 0.0),
     (418, 0.0),
     (419, 0.024060714060476288),
     (420, 0.0),
     (421, 0.012764224459658967),
     (422, 0.0),
     (423, 0.0),
     (424, 0.022899315106144215),
     (425, 0.016982139718431606),
     (426, 0.014998623088502793),
     (427, 0.0),
     (428, 0.2878505467001694),
     (429, 0.006147434170676051),
     (430, 0.004706180900561091),
     (431, 0.0),
     (432, 0.0),
     (433, 0.0),
     (434, 0.0),
     (435, 0.006966363756456715),
     (436, 0.004994615362565935),
     (437, 0.0),
     (438, 0.0),
     (439, 0.013518445110818376),
     (440, 0.015513391905168357),
     (441, 0.0),
     (442, 0.04305724012803792),
     (443, 0.0),
     (444, 0.0),
     (445, 0.0),
     (446, 0.0),
     (447, 0.009647569118333048),
     (448, 0.00878557520552861),
     (449, 0.047020755931188395),
     (450, 0.004225953101122851),
     (451, 0.0),
     (452, 0.0),
     (453, 0.011176868121732216),
     (454, 0.0),
     (455, 0.0),
     (456, 0.014954868857331733),
     (457, 0.0),
     (458, 0.0),
     (459, 0.0),
     (460, 0.0),
     (461, 0.0),
     (462, 0.0),
     (463, 0.0),
     (464, 0.0),
     (465, 0.006851991767686718),
     (466, 0.010839258474167463),
     (467, 0.0),
     (468, 0.0),
     (469, 0.03220449236430262),
     (470, 0.0),
     (471, 0.0),
     (472, 0.028267591905486646),
     (473, 0.0),
     (474, 0.0),
     (475, 0.0),
     (476, 0.010957912907903211),
     (477, 0.0),
     (478, 0.012876722207052398),
     (479, 0.004811403887831713),
     (480, 0.0),
     (481, 0.0),
     (482, 0.0),
     (483, 0.0),
     (484, 0.0),
     (485, 0.018839787404056257),
     (486, 0.028538272319874755),
     (487, 0.0),
     (488, 0.0050407188994071525),
     (489, 0.0),
     (490, 0.0),
     (491, 0.0),
     (492, 0.0),
     (493, 0.0),
     (494, 0.0),
     (495, 0.0),
     (496, 0.0),
     (497, 0.006215105570904383),
     (498, 0.0),
     (499, 0.0),
     (500, 0.010422600856505133),
     (501, 0.0),
     (502, 0.030069399054761754),
     (503, 0.014434423550061588),
     (504, 0.0),
     (505, 0.0),
     (506, 0.009754079225827956),
     (507, 0.0),
     (508, 0.005969841813625509),
     (509, 0.0045188912633002416),
     (510, 0.0),
     (511, 0.028422972525677374),
     (512, 0.0),
     (513, 0.013781537754424722),
     (514, 0.0),
     (515, 0.0),
     (516, 0.0),
     (517, 0.0),
     (518, 0.004235768670696995),
     (519, 0.0),
     (520, 0.0),
     (521, 0.0),
     (522, 0.0),
     (523, 0.0),
     (524, 0.0),
     (525, 0.0),
     (526, 0.0),
     (527, 0.0),
     (528, 0.01994770294983638),
     (529, 0.03718436565255274),
     (530, 0.007674139036513077),
     (531, 0.00931786513621838),
     (532, 0.0),
     (533, 0.0),
     (534, 0.0),
     (535, 0.0),
     (536, 0.0),
     (537, 0.0),
     (538, 0.014682417305252892),
     (539, 0.0),
     (540, 0.004321661832351155),
     (541, 0.00896585156811301),
     (542, 0.0),
     (543, 0.0),
     (544, 0.0),
     (545, 0.012211661956258339),
     (546, 0.0),
     (547, 0.0),
     (548, 0.0),
     (549, 0.026051576649055583),
     (550, 0.010283799766484835),
     (551, 0.014223150287082374),
     (552, 0.0),
     (553, 0.0),
     (554, 0.0),
     (555, 0.0),
     (556, 0.005584749797081471),
     (557, 0.0),
     (558, 0.0),
     (559, 0.0),
     (560, 0.0),
     (561, 0.03133684618426433),
     (562, 0.0),
     (563, 0.0),
     (564, 0.01943196273108645),
     (565, 0.0),
     (566, 0.0),
     (567, 0.0040999205319018186),
     (568, 0.0),
     (569, 0.030762295493311047),
     (570, 0.02245269395357705),
     (571, 0.0),
     (572, 0.0),
     (573, 0.0),
     (574, 0.031378939243315196),
     (575, 0.009776595843078384),
     (576, 0.04045920236760754),
     (577, 0.0),
     (578, 0.0),
     (579, 0.0),
     (580, 0.016935725113926287),
     (581, 0.0),
     (582, 0.02670926950147553),
     (583, 0.02490723981225111),
     (584, 0.0),
     (585, 0.0),
     (586, 0.015461612531061583),
     (587, 0.011376202068396842),
     (588, 0.009197594805883636),
     (589, 0.03841846242882966),
     (590, 0.06691123930989949),
     (591, 0.0),
     (592, 0.0),
     (593, 0.0),
     (594, 0.0),
     (595, 0.0),
     (596, 0.0),
     (597, 0.008797930425405939),
     (598, 0.03579469367796652),
     (599, 0.009341959621773555),
     (600, 0.014250603497467729),
     (601, 0.0057200893821864425),
     (602, 0.052575892836462385),
     (603, 0.0),
     (604, 0.0),
     (605, 0.01865065854523476),
     (606, 0.016897525589996614),
     (607, 0.027482933413845077),
     (608, 0.0),
     (609, 0.0),
     (610, 0.0),
     (611, 0.016917883999943602),
     (612, 0.00902470305981481),
     (613, 0.0),
     (614, 0.06410355106132812),
     (615, 0.04963450161220298),
     (616, 0.0),
     (617, 0.0),
     (618, 0.0),
     (619, 0.0),
     (620, 0.0),
     (621, 0.0),
     (622, 0.013903627921359758),
     (623, 0.0),
     (624, 0.0),
     (625, 0.0),
     (626, 0.0),
     (627, 0.02520094193815129),
     (628, 0.02048213640478558),
     (629, 0.004702362567008691),
     (630, 0.005052717064910142),
     (631, 0.0),
     (632, 0.0204743120399674),
     (633, 0.0),
     (634, 0.0),
     (635, 0.0),
     (636, 0.0),
     (637, 0.0),
     (638, 0.0),
     (639, 0.0),
     (640, 0.007150796354779989),
     (641, 0.0),
     (642, 0.0),
     (643, 0.00855517025853795),
     (644, 0.0),
     (645, 0.0),
     (646, 0.0),
     (647, 0.06113184684574714),
     (648, 0.0),
     (649, 0.0),
     (650, 0.0),
     (651, 0.0),
     (652, 0.012160362085977185),
     (653, 0.0),
     (654, 0.006422094020434747),
     (655, 0.0),
     (656, 0.0),
     (657, 0.01353065838248238),
     (658, 0.013202309107497236),
     (659, 0.0),
     (660, 0.0),
     (661, 0.0),
     (662, 0.005238449712920694),
     (663, 0.007007058524071615),
     (664, 0.008259685597076554),
     (665, 0.0),
     (666, 0.00982358881287198),
     (667, 0.0),
     (668, 0.057651269727644235),
     (669, 0.03314085888342038),
     (670, 0.020286710194779355),
     (671, 0.0),
     (672, 0.05212577297466591),
     (673, 0.005665395102159484),
     (674, 0.032584109983878305),
     (675, 0.0),
     (676, 0.0),
     (677, 0.0),
     (678, 0.0),
     (679, 0.006179926233101102),
     (680, 0.0),
     (681, 0.0),
     (682, 0.0),
     (683, 0.0),
     (684, 0.0),
     (685, 0.0),
     (686, 0.0),
     (687, 0.0),
     (688, 0.0),
     (689, 0.0),
     (690, 0.018989364459748505),
     (691, 0.005655903442126628),
     (692, 0.0),
     (693, 0.0),
     (694, 0.02961695482762623),
     (695, 0.03470603925033839),
     (696, 0.007905217080311212),
     (697, 0.0),
     (698, 0.015537488782031792),
     (699, 0.0),
     (700, 0.0),
     (701, 0.0),
     (702, 0.0),
     (703, 0.0),
     (704, 0.0),
     (705, 0.0),
     (706, 0.0),
     (707, 0.0),
     (708, 0.00947801222121816),
     (709, 0.0),
     (710, 0.0),
     (711, 0.0),
     (712, 0.010236171784831043),
     (713, 0.0),
     (714, 0.0),
     (715, 0.0),
     (716, 0.0),
     (717, 0.030195085274536487),
     (718, 0.0),
     (719, 0.0),
     (720, 0.0),
     (721, 0.007006637115952757),
     (722, 0.0),
     (723, 0.0),
     (724, 0.023029114239389456),
     (725, 0.0),
     (726, 0.005254585246277527),
     (727, 0.0),
     (728, 0.0),
     (729, 0.0),
     (730, 0.021472754599151447),
     (731, 0.022146305256816077),
     (732, 0.009444399972728227),
     (733, 0.0),
     (734, 0.022453930263541626),
     (735, 0.0),
     (736, 0.0),
     (737, 0.033617480380536996),
     (738, 0.0),
     (739, 0.0),
     (740, 0.01840119527534042),
     (741, 0.0),
     (742, 0.0),
     (743, 0.0),
     (744, 0.0),
     (745, 0.005553720811362761),
     (746, 0.0),
     (747, 0.008496139108886122),
     (748, 0.0),
     (749, 0.0),
     (750, 0.0),
     (751, 0.0),
     (752, 0.01235569253813857),
     (753, 0.0),
     (754, 0.0),
     (755, 0.025754701125838345),
     (756, 0.0),
     (757, 0.0),
     (758, 0.011111953707018762),
     (759, 0.009551545459859773),
     (760, 0.0),
     (761, 0.040652701400888086),
     (762, 0.007796056413369449),
     (763, 0.008663354895634594),
     (764, 0.011002378984487964),
     (765, 0.0),
     (766, 0.0),
     (767, 0.0),
     (768, 0.0),
     (769, 0.0),
     (770, 0.00854504798385258),
     (771, 0.051989545802567004),
     (772, 0.0),
     (773, 0.0),
     (774, 0.0),
     (775, 0.010802681598897336),
     (776, 0.0),
     (777, 0.06007980793346218),
     (778, 0.007170458690441402),
     (779, 0.015245689486302672),
     (780, 0.0),
     (781, 0.0),
     (782, 0.03750215651562471),
     (783, 0.0),
     (784, 0.013870497592700797),
     (785, 0.014192238901799),
     (786, 0.007164362933647063),
     (787, 0.022466076220887043),
     (788, 0.01645735229132369),
     (789, 0.0),
     (790, 0.08025680374928534),
     (791, 0.0),
     (792, 0.005030442812173936),
     (793, 0.0),
     (794, 0.0),
     (795, 0.0),
     (796, 0.0),
     (797, 0.03848369401874538),
     (798, 0.0),
     (799, 0.021954259841651464),
     (800, 0.0),
     (801, 0.01112348598061646),
     (802, 0.034745364808828505),
     (803, 0.022204492708220076),
     (804, 0.0),
     (805, 0.038251511379121826),
     (806, 0.0),
     (807, 0.04172310532275985),
     (808, 0.0),
     (809, 0.0),
     (810, 0.019379625179082663),
     (811, 0.0),
     (812, 0.005655417665942667),
     (813, 0.0),
     (814, 0.0),
     (815, 0.0),
     (816, 0.007462989531691867),
     (817, 0.0),
     (818, 0.009507123465255158),
     (819, 0.0),
     (820, 0.04298997885426019),
     (821, 0.01028912795689598),
     (822, 0.0),
     (823, 0.0),
     (824, 0.0),
     (825, 0.0),
     (826, 0.0),
     (827, 0.0),
     (828, 0.012620623610056555),
     (829, 0.0),
     (830, 0.011821011980776995),
     (831, 0.0),
     (832, 0.0),
     (833, 0.016274908639033928),
     (834, 0.016419770858221914),
     (835, 0.005140249250759681),
     (836, 0.0),
     (837, 0.0),
     (838, 0.0),
     (839, 0.0),
     (840, 0.0),
     (841, 0.0),
     (842, 0.0160278869432615),
     (843, 0.0),
     (844, 0.0),
     (845, 0.005074194504145629),
     (846, 0.0),
     (847, 0.008626499485339136),
     (848, 0.0),
     (849, 0.014856684811016888),
     (850, 0.0),
     (851, 0.0),
     (852, 0.0),
     (853, 0.0),
     (854, 0.04387394985319228),
     (855, 0.0),
     (856, 0.0),
     (857, 0.0),
     (858, 0.0),
     (859, 0.0),
     (860, 0.0),
     (861, 0.009104905195773793),
     (862, 0.0),
     (863, 0.0),
     (864, 0.0),
     (865, 0.0),
     (866, 0.031967047989104),
     (867, 0.00642858038077249),
     (868, 0.005100417269332049),
     (869, 0.0),
     (870, 0.031044854138300017),
     (871, 0.0),
     (872, 0.0),
     (873, 0.03252808053782304),
     (874, 0.01453908247211661),
     (875, 0.0),
     (876, 0.012832147519280471),
     (877, 0.0),
     (878, 0.0),
     (879, 0.1050189187666832),
     (880, 0.0),
     (881, 0.0),
     (882, 0.0),
     (883, 0.0),
     (884, 0.05513869660012231),
     (885, 0.0),
     (886, 0.0),
     (887, 0.005494342719097118),
     (888, 0.0),
     (889, 0.0),
     (890, 0.0),
     (891, 0.0),
     (892, 0.02987487486390953),
     (893, 0.0),
     (894, 0.01874742586999907),
     (895, 0.0),
     (896, 0.0),
     (897, 0.0),
     (898, 0.0),
     (899, 0.0),
     (900, 0.006709804958300454),
     (901, 0.0),
     (902, 0.0),
     (903, 0.0),
     (904, 0.01943066140914914),
     (905, 0.0),
     (906, 0.007476890926395708),
     (907, 0.014636581512875103),
     (908, 0.0),
     (909, 0.04744169902641941),
     (910, 0.0037213786892250312),
     (911, 0.0),
     (912, 0.025633927478456803),
     (913, 0.0),
     (914, 0.0),
     (915, 0.0),
     (916, 0.0),
     (917, 0.0),
     (918, 0.009753806176412302),
     (919, 0.014053662451009762),
     (920, 0.012136900418604596),
     (921, 0.0),
     (922, 0.0),
     (923, 0.0),
     (924, 0.0),
     (925, 0.0),
     (926, 0.0),
     (927, 0.0),
     (928, 0.0),
     (929, 0.0),
     (930, 0.014377375158525135),
     (931, 0.0),
     (932, 0.013184032702353046),
     (933, 0.0),
     (934, 0.025653976336290716),
     (935, 0.00643170935527381),
     (936, 0.0),
     (937, 0.027133916072073248),
     (938, 0.0),
     (939, 0.012517799110931985),
     (940, 0.0),
     (941, 0.0),
     (942, 0.022362945890286406),
     (943, 0.0312711877499085),
     (944, 0.0),
     (945, 0.0),
     (946, 0.0),
     (947, 0.0),
     (948, 0.0),
     (949, 0.0057020914145596855),
     (950, 0.010320635579054792),
     (951, 0.0),
     (952, 0.0),
     (953, 0.004779570290121787),
     (954, 0.0),
     (955, 0.0),
     (956, 0.018135294954648988),
     (957, 0.007031658623135218),
     (958, 0.023758468315152943),
     (959, 0.0),
     (960, 0.0),
     (961, 0.0),
     (962, 0.03453964161915732),
     (963, 0.0),
     (964, 0.012491386667189571),
     (965, 0.0),
     (966, 0.0),
     (967, 0.015790411621549087),
     (968, 0.04504721127283797),
     (969, 0.0),
     (970, 0.0),
     (971, 0.016492216563284513),
     (972, 0.0),
     (973, 0.0),
     (974, 0.0),
     (975, 0.05123326179177315),
     (976, 0.021712405104773222),
     (977, 0.0),
     (978, 0.005058532807693165),
     (979, 0.0810447942804151),
     (980, 0.01831948321713123),
     (981, 0.03276074913231903),
     (982, 0.012718170351844353),
     (983, 0.0),
     (984, 0.0),
     (985, 0.02749128951705554),
     (986, 0.0),
     (987, 0.010139537004863123),
     (988, 0.0),
     (989, 0.0),
     (990, 0.0),
     (991, 0.0038287385710778178),
     (992, 0.05181241244828169),
     (993, 0.0),
     (994, 0.0),
     (995, 0.0),
     (996, 0.0),
     (997, 0.0),
     (998, 0.0),
     (999, 0.0),
     ...]




```python
test_sim_scores = sorted(test_sim_scores, key=lambda x: x[1], reverse=True) # 코사인 유사도 기준으로 내림차순 정렬
test_sim_scores[1:11] # 자기 자신을 제외한 10개의 추천 영화를 슬라이싱
```




    [(65, 0.30151176591665485),
     (299, 0.29857045255396825),
     (428, 0.2878505467001694),
     (1359, 0.264460923827995),
     (3854, 0.18545003006561456),
     (119, 0.16799626199850706),
     (2507, 0.16682891043358278),
     (9, 0.1337400906655523),
     (1181, 0.13219702138476813),
     (210, 0.13045537014449818)]



lambda식 설명


```python
def get_second(x):
    return x[1]

lst = ['인덱스', '유사도']
print(get_second(lst))
```

    유사도
    


```python
(lambda x: x[1])(lst)
```




    '유사도'




```python
# 추천 영화 목록 10개의 인덱스 정보 추출
test_movie_indices = [i[0] for i in test_sim_scores[1:11]]
test_movie_indices
```




    [65, 299, 428, 1359, 3854, 119, 2507, 9, 1181, 210]




```python
# 인덱스 정보를 통해 영화 제목 추출
df2['title'].iloc[test_movie_indices]
```




    65                              The Dark Knight
    299                              Batman Forever
    428                              Batman Returns
    1359                                     Batman
    3854    Batman: The Dark Knight Returns, Part 2
    119                               Batman Begins
    2507                                  Slow Burn
    9            Batman v Superman: Dawn of Justice
    1181                                        JFK
    210                              Batman & Robin
    Name: title, dtype: object




```python
df2['title'][:20]
```




    0                                          Avatar
    1        Pirates of the Caribbean: At World's End
    2                                         Spectre
    3                           The Dark Knight Rises
    4                                     John Carter
    5                                    Spider-Man 3
    6                                         Tangled
    7                         Avengers: Age of Ultron
    8          Harry Potter and the Half-Blood Prince
    9              Batman v Superman: Dawn of Justice
    10                               Superman Returns
    11                              Quantum of Solace
    12     Pirates of the Caribbean: Dead Man's Chest
    13                                The Lone Ranger
    14                                   Man of Steel
    15       The Chronicles of Narnia: Prince Caspian
    16                                   The Avengers
    17    Pirates of the Caribbean: On Stranger Tides
    18                                 Men in Black 3
    19      The Hobbit: The Battle of the Five Armies
    Name: title, dtype: object



get_recommendations 함수 호출


```python
get_recommendations('Avengers: Age of Ultron')
```




    16                    The Avengers
    79                      Iron Man 2
    68                        Iron Man
    26      Captain America: Civil War
    227                 Knight and Day
    31                      Iron Man 3
    1868            Cradle 2 the Grave
    344                    Unstoppable
    1922                    Gettysburg
    531        The Man from U.N.C.L.E.
    Name: title, dtype: object




```python
get_recommendations('The Avengers')
```




    7               Avengers: Age of Ultron
    3144                            Plastic
    1715                            Timecop
    4124                 This Thing of Ours
    3311              Thank You for Smoking
    3033                      The Corruptor
    588     Wall Street: Money Never Sleeps
    2136         Team America: World Police
    1468                       The Fountain
    1286                        Snowpiercer
    Name: title, dtype: object



### 다양한 요소 기반 추천 (장르, 감독, 키워드 등)


```python
df2.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>keywords</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>popularity</th>
      <th>production_companies</th>
      <th>...</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>cast</th>
      <th>crew</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>237000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://www.avatarmovie.com/</td>
      <td>19995</td>
      <td>[{"id": 1463, "name": "culture clash"}, {"id":...</td>
      <td>en</td>
      <td>Avatar</td>
      <td>In the 22nd century, a paraplegic Marine is di...</td>
      <td>150.437577</td>
      <td>[{"name": "Ingenious Film Partners", "id": 289...</td>
      <td>...</td>
      <td>2787965087</td>
      <td>162.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}, {"iso...</td>
      <td>Released</td>
      <td>Enter the World of Pandora.</td>
      <td>Avatar</td>
      <td>7.2</td>
      <td>11800</td>
      <td>[{"cast_id": 242, "character": "Jake Sully", "...</td>
      <td>[{"credit_id": "52fe48009251416c750aca23", "de...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>300000000</td>
      <td>[{"id": 12, "name": "Adventure"}, {"id": 14, "...</td>
      <td>http://disney.go.com/disneypictures/pirates/</td>
      <td>285</td>
      <td>[{"id": 270, "name": "ocean"}, {"id": 726, "na...</td>
      <td>en</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>Captain Barbossa, long believed to be dead, ha...</td>
      <td>139.082615</td>
      <td>[{"name": "Walt Disney Pictures", "id": 2}, {"...</td>
      <td>...</td>
      <td>961000000</td>
      <td>169.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>At the end of the world, the adventure begins.</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>6.9</td>
      <td>4500</td>
      <td>[{"cast_id": 4, "character": "Captain Jack Spa...</td>
      <td>[{"credit_id": "52fe4232c3a36847f800b579", "de...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>245000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://www.sonypictures.com/movies/spectre/</td>
      <td>206647</td>
      <td>[{"id": 470, "name": "spy"}, {"id": 818, "name...</td>
      <td>en</td>
      <td>Spectre</td>
      <td>A cryptic message from Bond’s past sends him o...</td>
      <td>107.376788</td>
      <td>[{"name": "Columbia Pictures", "id": 5}, {"nam...</td>
      <td>...</td>
      <td>880674609</td>
      <td>148.0</td>
      <td>[{"iso_639_1": "fr", "name": "Fran\u00e7ais"},...</td>
      <td>Released</td>
      <td>A Plan No One Escapes</td>
      <td>Spectre</td>
      <td>6.3</td>
      <td>4466</td>
      <td>[{"cast_id": 1, "character": "James Bond", "cr...</td>
      <td>[{"credit_id": "54805967c3a36829b5002c41", "de...</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 22 columns</p>
</div>




```python
df2.loc[0, 'genres'] #0행, genres열의 데이터를 불러옴
```




    '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'



Example)


```python
s1 = [{"id": 28, "name": "Action"}]
s2 = '[{"id": 28, "name": "Action"}]'
```


```python
type(s1), type(s2)
```




    (list, str)



데이터 전처리(스트링으로된 데이터를 List로 바뀌주는 작업)


```python
from ast import literal_eval
s2 = literal_eval(s2)
s2, type(s2)
```




    ([{'id': 28, 'name': 'Action'}], list)




```python
print(s1)
print(s2)
```

    [{'id': 28, 'name': 'Action'}]
    [{'id': 28, 'name': 'Action'}]
    


```python
features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(literal_eval)
```


```python
df2.loc[0, 'crew']
```




    [{'credit_id': '52fe48009251416c750aca23',
      'department': 'Editing',
      'gender': 0,
      'id': 1721,
      'job': 'Editor',
      'name': 'Stephen E. Rivkin'},
     {'credit_id': '539c47ecc3a36810e3001f87',
      'department': 'Art',
      'gender': 2,
      'id': 496,
      'job': 'Production Design',
      'name': 'Rick Carter'},
     {'credit_id': '54491c89c3a3680fb4001cf7',
      'department': 'Sound',
      'gender': 0,
      'id': 900,
      'job': 'Sound Designer',
      'name': 'Christopher Boyes'},
     {'credit_id': '54491cb70e0a267480001bd0',
      'department': 'Sound',
      'gender': 0,
      'id': 900,
      'job': 'Supervising Sound Editor',
      'name': 'Christopher Boyes'},
     {'credit_id': '539c4a4cc3a36810c9002101',
      'department': 'Production',
      'gender': 1,
      'id': 1262,
      'job': 'Casting',
      'name': 'Mali Finn'},
     {'credit_id': '5544ee3b925141499f0008fc',
      'department': 'Sound',
      'gender': 2,
      'id': 1729,
      'job': 'Original Music Composer',
      'name': 'James Horner'},
     {'credit_id': '52fe48009251416c750ac9c3',
      'department': 'Directing',
      'gender': 2,
      'id': 2710,
      'job': 'Director',
      'name': 'James Cameron'},
     {'credit_id': '52fe48009251416c750ac9d9',
      'department': 'Writing',
      'gender': 2,
      'id': 2710,
      'job': 'Writer',
      'name': 'James Cameron'},
     {'credit_id': '52fe48009251416c750aca17',
      'department': 'Editing',
      'gender': 2,
      'id': 2710,
      'job': 'Editor',
      'name': 'James Cameron'},
     {'credit_id': '52fe48009251416c750aca29',
      'department': 'Production',
      'gender': 2,
      'id': 2710,
      'job': 'Producer',
      'name': 'James Cameron'},
     {'credit_id': '52fe48009251416c750aca3f',
      'department': 'Writing',
      'gender': 2,
      'id': 2710,
      'job': 'Screenplay',
      'name': 'James Cameron'},
     {'credit_id': '539c4987c3a36810ba0021a4',
      'department': 'Art',
      'gender': 2,
      'id': 7236,
      'job': 'Art Direction',
      'name': 'Andrew Menzies'},
     {'credit_id': '549598c3c3a3686ae9004383',
      'department': 'Visual Effects',
      'gender': 0,
      'id': 6690,
      'job': 'Visual Effects Producer',
      'name': 'Jill Brooks'},
     {'credit_id': '52fe48009251416c750aca4b',
      'department': 'Production',
      'gender': 1,
      'id': 6347,
      'job': 'Casting',
      'name': 'Margery Simkin'},
     {'credit_id': '570b6f419251417da70032fe',
      'department': 'Art',
      'gender': 2,
      'id': 6878,
      'job': 'Supervising Art Director',
      'name': 'Kevin Ishioka'},
     {'credit_id': '5495a0fac3a3686ae9004468',
      'department': 'Sound',
      'gender': 0,
      'id': 6883,
      'job': 'Music Editor',
      'name': 'Dick Bernstein'},
     {'credit_id': '54959706c3a3686af3003e81',
      'department': 'Sound',
      'gender': 0,
      'id': 8159,
      'job': 'Sound Effects Editor',
      'name': 'Shannon Mills'},
     {'credit_id': '54491d58c3a3680fb1001ccb',
      'department': 'Sound',
      'gender': 0,
      'id': 8160,
      'job': 'Foley',
      'name': 'Dennie Thorpe'},
     {'credit_id': '54491d6cc3a3680fa5001b2c',
      'department': 'Sound',
      'gender': 0,
      'id': 8163,
      'job': 'Foley',
      'name': 'Jana Vance'},
     {'credit_id': '52fe48009251416c750aca57',
      'department': 'Costume & Make-Up',
      'gender': 1,
      'id': 8527,
      'job': 'Costume Design',
      'name': 'Deborah Lynn Scott'},
     {'credit_id': '52fe48009251416c750aca2f',
      'department': 'Production',
      'gender': 2,
      'id': 8529,
      'job': 'Producer',
      'name': 'Jon Landau'},
     {'credit_id': '539c4937c3a36810ba002194',
      'department': 'Art',
      'gender': 0,
      'id': 9618,
      'job': 'Art Direction',
      'name': 'Sean Haworth'},
     {'credit_id': '539c49b6c3a36810c10020e6',
      'department': 'Art',
      'gender': 1,
      'id': 12653,
      'job': 'Set Decoration',
      'name': 'Kim Sinclair'},
     {'credit_id': '570b6f2f9251413a0e00020d',
      'department': 'Art',
      'gender': 1,
      'id': 12653,
      'job': 'Supervising Art Director',
      'name': 'Kim Sinclair'},
     {'credit_id': '54491a6c0e0a26748c001b19',
      'department': 'Art',
      'gender': 2,
      'id': 14350,
      'job': 'Set Designer',
      'name': 'Richard F. Mays'},
     {'credit_id': '56928cf4c3a3684cff0025c4',
      'department': 'Production',
      'gender': 1,
      'id': 20294,
      'job': 'Executive Producer',
      'name': 'Laeta Kalogridis'},
     {'credit_id': '52fe48009251416c750aca51',
      'department': 'Costume & Make-Up',
      'gender': 0,
      'id': 17675,
      'job': 'Costume Design',
      'name': 'Mayes C. Rubeo'},
     {'credit_id': '52fe48009251416c750aca11',
      'department': 'Camera',
      'gender': 2,
      'id': 18265,
      'job': 'Director of Photography',
      'name': 'Mauro Fiore'},
     {'credit_id': '5449194d0e0a26748f001b39',
      'department': 'Art',
      'gender': 0,
      'id': 42281,
      'job': 'Set Designer',
      'name': 'Scott Herbertson'},
     {'credit_id': '52fe48009251416c750aca05',
      'department': 'Crew',
      'gender': 0,
      'id': 42288,
      'job': 'Stunts',
      'name': 'Woody Schultz'},
     {'credit_id': '5592aefb92514152de0010f5',
      'department': 'Costume & Make-Up',
      'gender': 0,
      'id': 29067,
      'job': 'Makeup Artist',
      'name': 'Linda DeVetta'},
     {'credit_id': '5592afa492514152de00112c',
      'department': 'Costume & Make-Up',
      'gender': 0,
      'id': 29067,
      'job': 'Hairstylist',
      'name': 'Linda DeVetta'},
     {'credit_id': '54959ed592514130fc002e5d',
      'department': 'Camera',
      'gender': 2,
      'id': 33302,
      'job': 'Camera Operator',
      'name': 'Richard Bluck'},
     {'credit_id': '539c4891c3a36810ba002147',
      'department': 'Art',
      'gender': 2,
      'id': 33303,
      'job': 'Art Direction',
      'name': 'Simon Bright'},
     {'credit_id': '54959c069251417a81001f3a',
      'department': 'Visual Effects',
      'gender': 0,
      'id': 113145,
      'job': 'Visual Effects Supervisor',
      'name': 'Richard Martin'},
     {'credit_id': '54959a0dc3a3680ff5002c8d',
      'department': 'Crew',
      'gender': 2,
      'id': 58188,
      'job': 'Visual Effects Editor',
      'name': 'Steve R. Moore'},
     {'credit_id': '52fe48009251416c750aca1d',
      'department': 'Editing',
      'gender': 2,
      'id': 58871,
      'job': 'Editor',
      'name': 'John Refoua'},
     {'credit_id': '54491a4dc3a3680fc30018ca',
      'department': 'Art',
      'gender': 0,
      'id': 92359,
      'job': 'Set Designer',
      'name': 'Karl J. Martin'},
     {'credit_id': '52fe48009251416c750aca35',
      'department': 'Camera',
      'gender': 1,
      'id': 72201,
      'job': 'Director of Photography',
      'name': 'Chiling Lin'},
     {'credit_id': '52fe48009251416c750ac9ff',
      'department': 'Crew',
      'gender': 0,
      'id': 89714,
      'job': 'Stunts',
      'name': 'Ilram Choi'},
     {'credit_id': '54959c529251416e2b004394',
      'department': 'Visual Effects',
      'gender': 2,
      'id': 93214,
      'job': 'Visual Effects Supervisor',
      'name': 'Steven Quale'},
     {'credit_id': '54491edf0e0a267489001c37',
      'department': 'Crew',
      'gender': 1,
      'id': 122607,
      'job': 'Dialect Coach',
      'name': 'Carla Meyer'},
     {'credit_id': '539c485bc3a368653d001a3a',
      'department': 'Art',
      'gender': 2,
      'id': 132585,
      'job': 'Art Direction',
      'name': 'Nick Bassett'},
     {'credit_id': '539c4903c3a368653d001a74',
      'department': 'Art',
      'gender': 0,
      'id': 132596,
      'job': 'Art Direction',
      'name': 'Jill Cormack'},
     {'credit_id': '539c4967c3a368653d001a94',
      'department': 'Art',
      'gender': 0,
      'id': 132604,
      'job': 'Art Direction',
      'name': 'Andy McLaren'},
     {'credit_id': '52fe48009251416c750aca45',
      'department': 'Crew',
      'gender': 0,
      'id': 236696,
      'job': 'Motion Capture Artist',
      'name': 'Terry Notary'},
     {'credit_id': '54959e02c3a3680fc60027d2',
      'department': 'Crew',
      'gender': 2,
      'id': 956198,
      'job': 'Stunt Coordinator',
      'name': 'Garrett Warren'},
     {'credit_id': '54959ca3c3a3686ae300438c',
      'department': 'Visual Effects',
      'gender': 2,
      'id': 957874,
      'job': 'Visual Effects Supervisor',
      'name': 'Jonathan Rothbart'},
     {'credit_id': '570b6f519251412c74001b2f',
      'department': 'Art',
      'gender': 0,
      'id': 957889,
      'job': 'Supervising Art Director',
      'name': 'Stefan Dechant'},
     {'credit_id': '570b6f62c3a3680b77007460',
      'department': 'Art',
      'gender': 2,
      'id': 959555,
      'job': 'Supervising Art Director',
      'name': 'Todd Cherniawsky'},
     {'credit_id': '539c4a3ac3a36810da0021cc',
      'department': 'Production',
      'gender': 0,
      'id': 1016177,
      'job': 'Casting',
      'name': 'Miranda Rivers'},
     {'credit_id': '539c482cc3a36810c1002062',
      'department': 'Art',
      'gender': 0,
      'id': 1032536,
      'job': 'Production Design',
      'name': 'Robert Stromberg'},
     {'credit_id': '539c4b65c3a36810c9002125',
      'department': 'Costume & Make-Up',
      'gender': 2,
      'id': 1071680,
      'job': 'Costume Design',
      'name': 'John Harding'},
     {'credit_id': '54959e6692514130fc002e4e',
      'department': 'Camera',
      'gender': 0,
      'id': 1177364,
      'job': 'Steadicam Operator',
      'name': 'Roberto De Angelis'},
     {'credit_id': '539c49f1c3a368653d001aac',
      'department': 'Costume & Make-Up',
      'gender': 2,
      'id': 1202850,
      'job': 'Makeup Department Head',
      'name': 'Mike Smithson'},
     {'credit_id': '5495999ec3a3686ae100460c',
      'department': 'Visual Effects',
      'gender': 0,
      'id': 1204668,
      'job': 'Visual Effects Producer',
      'name': 'Alain Lalanne'},
     {'credit_id': '54959cdfc3a3681153002729',
      'department': 'Visual Effects',
      'gender': 0,
      'id': 1206410,
      'job': 'Visual Effects Supervisor',
      'name': 'Lucas Salton'},
     {'credit_id': '549596239251417a81001eae',
      'department': 'Crew',
      'gender': 0,
      'id': 1234266,
      'job': 'Post Production Supervisor',
      'name': 'Janace Tashjian'},
     {'credit_id': '54959c859251416e1e003efe',
      'department': 'Visual Effects',
      'gender': 0,
      'id': 1271932,
      'job': 'Visual Effects Supervisor',
      'name': 'Stephen Rosenbaum'},
     {'credit_id': '5592af28c3a368775a00105f',
      'department': 'Costume & Make-Up',
      'gender': 0,
      'id': 1310064,
      'job': 'Makeup Artist',
      'name': 'Frankie Karena'},
     {'credit_id': '539c4adfc3a36810e300203b',
      'department': 'Costume & Make-Up',
      'gender': 1,
      'id': 1319844,
      'job': 'Costume Supervisor',
      'name': 'Lisa Lovaas'},
     {'credit_id': '54959b579251416e2b004371',
      'department': 'Visual Effects',
      'gender': 0,
      'id': 1327028,
      'job': 'Visual Effects Supervisor',
      'name': 'Jonathan Fawkner'},
     {'credit_id': '539c48a7c3a36810b5001fa7',
      'department': 'Art',
      'gender': 0,
      'id': 1330561,
      'job': 'Art Direction',
      'name': 'Robert Bavin'},
     {'credit_id': '539c4a71c3a36810da0021e0',
      'department': 'Costume & Make-Up',
      'gender': 0,
      'id': 1330567,
      'job': 'Costume Supervisor',
      'name': 'Anthony Almaraz'},
     {'credit_id': '539c4a8ac3a36810ba0021e4',
      'department': 'Costume & Make-Up',
      'gender': 0,
      'id': 1330570,
      'job': 'Costume Supervisor',
      'name': 'Carolyn M. Fenton'},
     {'credit_id': '539c4ab6c3a36810da0021f0',
      'department': 'Costume & Make-Up',
      'gender': 0,
      'id': 1330574,
      'job': 'Costume Supervisor',
      'name': 'Beth Koenigsberg'},
     {'credit_id': '54491ab70e0a267480001ba2',
      'department': 'Art',
      'gender': 0,
      'id': 1336191,
      'job': 'Set Designer',
      'name': 'Sam Page'},
     {'credit_id': '544919d9c3a3680fc30018bd',
      'department': 'Art',
      'gender': 0,
      'id': 1339441,
      'job': 'Set Designer',
      'name': 'Tex Kadonaga'},
     {'credit_id': '54491cf50e0a267483001b0c',
      'department': 'Editing',
      'gender': 0,
      'id': 1352422,
      'job': 'Dialogue Editor',
      'name': 'Kim Foscato'},
     {'credit_id': '544919f40e0a26748c001b09',
      'department': 'Art',
      'gender': 0,
      'id': 1352962,
      'job': 'Set Designer',
      'name': 'Tammy S. Lee'},
     {'credit_id': '5495a115c3a3680ff5002d71',
      'department': 'Crew',
      'gender': 0,
      'id': 1357070,
      'job': 'Transportation Coordinator',
      'name': 'Denny Caira'},
     {'credit_id': '5495a12f92514130fc002e94',
      'department': 'Crew',
      'gender': 0,
      'id': 1357071,
      'job': 'Transportation Coordinator',
      'name': 'James Waitkus'},
     {'credit_id': '5495976fc3a36811530026b0',
      'department': 'Sound',
      'gender': 0,
      'id': 1360103,
      'job': 'Supervising Sound Editor',
      'name': 'Addison Teague'},
     {'credit_id': '54491837c3a3680fb1001c5a',
      'department': 'Art',
      'gender': 2,
      'id': 1376887,
      'job': 'Set Designer',
      'name': 'C. Scott Baker'},
     {'credit_id': '54491878c3a3680fb4001c9d',
      'department': 'Art',
      'gender': 0,
      'id': 1376888,
      'job': 'Set Designer',
      'name': 'Luke Caska'},
     {'credit_id': '544918dac3a3680fa5001ae0',
      'department': 'Art',
      'gender': 0,
      'id': 1376889,
      'job': 'Set Designer',
      'name': 'David Chow'},
     {'credit_id': '544919110e0a267486001b68',
      'department': 'Art',
      'gender': 0,
      'id': 1376890,
      'job': 'Set Designer',
      'name': 'Jonathan Dyer'},
     {'credit_id': '54491967c3a3680faa001b5e',
      'department': 'Art',
      'gender': 0,
      'id': 1376891,
      'job': 'Set Designer',
      'name': 'Joseph Hiura'},
     {'credit_id': '54491997c3a3680fb1001c8a',
      'department': 'Art',
      'gender': 0,
      'id': 1376892,
      'job': 'Art Department Coordinator',
      'name': 'Rebecca Jellie'},
     {'credit_id': '544919ba0e0a26748f001b42',
      'department': 'Art',
      'gender': 0,
      'id': 1376893,
      'job': 'Set Designer',
      'name': 'Robert Andrew Johnson'},
     {'credit_id': '54491b1dc3a3680faa001b8c',
      'department': 'Art',
      'gender': 0,
      'id': 1376895,
      'job': 'Assistant Art Director',
      'name': 'Mike Stassi'},
     {'credit_id': '54491b79c3a3680fbb001826',
      'department': 'Art',
      'gender': 0,
      'id': 1376897,
      'job': 'Construction Coordinator',
      'name': 'John Villarino'},
     {'credit_id': '54491baec3a3680fb4001ce6',
      'department': 'Art',
      'gender': 2,
      'id': 1376898,
      'job': 'Assistant Art Director',
      'name': 'Jeffrey Wisniewski'},
     {'credit_id': '54491d2fc3a3680fb4001d07',
      'department': 'Editing',
      'gender': 0,
      'id': 1376899,
      'job': 'Dialogue Editor',
      'name': 'Cheryl Nardi'},
     {'credit_id': '54491d86c3a3680fa5001b2f',
      'department': 'Editing',
      'gender': 0,
      'id': 1376901,
      'job': 'Dialogue Editor',
      'name': 'Marshall Winn'},
     {'credit_id': '54491d9dc3a3680faa001bb0',
      'department': 'Sound',
      'gender': 0,
      'id': 1376902,
      'job': 'Supervising Sound Editor',
      'name': 'Gwendolyn Yates Whittle'},
     {'credit_id': '54491dc10e0a267486001bce',
      'department': 'Sound',
      'gender': 0,
      'id': 1376903,
      'job': 'Sound Re-Recording Mixer',
      'name': 'William Stein'},
     {'credit_id': '54491f500e0a26747c001c07',
      'department': 'Crew',
      'gender': 0,
      'id': 1376909,
      'job': 'Choreographer',
      'name': 'Lula Washington'},
     {'credit_id': '549599239251412c4e002a2e',
      'department': 'Visual Effects',
      'gender': 0,
      'id': 1391692,
      'job': 'Visual Effects Producer',
      'name': 'Chris Del Conte'},
     {'credit_id': '54959d54c3a36831b8001d9a',
      'department': 'Visual Effects',
      'gender': 2,
      'id': 1391695,
      'job': 'Visual Effects Supervisor',
      'name': 'R. Christopher White'},
     {'credit_id': '54959bdf9251412c4e002a66',
      'department': 'Visual Effects',
      'gender': 0,
      'id': 1394070,
      'job': 'Visual Effects Supervisor',
      'name': 'Dan Lemmon'},
     {'credit_id': '5495971d92514132ed002922',
      'department': 'Sound',
      'gender': 0,
      'id': 1394129,
      'job': 'Sound Effects Editor',
      'name': 'Tim Nielsen'},
     {'credit_id': '5592b25792514152cc0011aa',
      'department': 'Crew',
      'gender': 0,
      'id': 1394286,
      'job': 'CG Supervisor',
      'name': 'Michael Mulholland'},
     {'credit_id': '54959a329251416e2b004355',
      'department': 'Crew',
      'gender': 0,
      'id': 1394750,
      'job': 'Visual Effects Editor',
      'name': 'Thomas Nittmann'},
     {'credit_id': '54959d6dc3a3686ae9004401',
      'department': 'Visual Effects',
      'gender': 0,
      'id': 1394755,
      'job': 'Visual Effects Supervisor',
      'name': 'Edson Williams'},
     {'credit_id': '5495a08fc3a3686ae300441c',
      'department': 'Editing',
      'gender': 0,
      'id': 1394953,
      'job': 'Digital Intermediate',
      'name': 'Christine Carr'},
     {'credit_id': '55402d659251413d6d000249',
      'department': 'Visual Effects',
      'gender': 0,
      'id': 1395269,
      'job': 'Visual Effects Supervisor',
      'name': 'John Bruno'},
     {'credit_id': '54959e7b9251416e1e003f3e',
      'department': 'Camera',
      'gender': 0,
      'id': 1398970,
      'job': 'Steadicam Operator',
      'name': 'David Emmerichs'},
     {'credit_id': '54959734c3a3686ae10045e0',
      'department': 'Sound',
      'gender': 0,
      'id': 1400906,
      'job': 'Sound Effects Editor',
      'name': 'Christopher Scarabosio'},
     {'credit_id': '549595dd92514130fc002d79',
      'department': 'Production',
      'gender': 0,
      'id': 1401784,
      'job': 'Production Supervisor',
      'name': 'Jennifer Teves'},
     {'credit_id': '549596009251413af70028cc',
      'department': 'Production',
      'gender': 0,
      'id': 1401785,
      'job': 'Production Manager',
      'name': 'Brigitte Yorke'},
     {'credit_id': '549596e892514130fc002d99',
      'department': 'Sound',
      'gender': 0,
      'id': 1401786,
      'job': 'Sound Effects Editor',
      'name': 'Ken Fischer'},
     {'credit_id': '549598229251412c4e002a1c',
      'department': 'Crew',
      'gender': 0,
      'id': 1401787,
      'job': 'Special Effects Coordinator',
      'name': 'Iain Hutton'},
     {'credit_id': '549598349251416e2b00432b',
      'department': 'Crew',
      'gender': 0,
      'id': 1401788,
      'job': 'Special Effects Coordinator',
      'name': 'Steve Ingram'},
     {'credit_id': '54959905c3a3686ae3004324',
      'department': 'Visual Effects',
      'gender': 0,
      'id': 1401789,
      'job': 'Visual Effects Producer',
      'name': 'Joyce Cox'},
     {'credit_id': '5495994b92514132ed002951',
      'department': 'Visual Effects',
      'gender': 0,
      'id': 1401790,
      'job': 'Visual Effects Producer',
      'name': 'Jenny Foster'},
     {'credit_id': '549599cbc3a3686ae1004613',
      'department': 'Crew',
      'gender': 0,
      'id': 1401791,
      'job': 'Visual Effects Editor',
      'name': 'Christopher Marino'},
     {'credit_id': '549599f2c3a3686ae100461e',
      'department': 'Crew',
      'gender': 0,
      'id': 1401792,
      'job': 'Visual Effects Editor',
      'name': 'Jim Milton'},
     {'credit_id': '54959a51c3a3686af3003eb5',
      'department': 'Visual Effects',
      'gender': 0,
      'id': 1401793,
      'job': 'Visual Effects Producer',
      'name': 'Cyndi Ochs'},
     {'credit_id': '54959a7cc3a36811530026f4',
      'department': 'Crew',
      'gender': 0,
      'id': 1401794,
      'job': 'Visual Effects Editor',
      'name': 'Lucas Putnam'},
     {'credit_id': '54959b91c3a3680ff5002cb4',
      'department': 'Visual Effects',
      'gender': 0,
      'id': 1401795,
      'job': 'Visual Effects Supervisor',
      'name': "Anthony 'Max' Ivins"},
     {'credit_id': '54959bb69251412c4e002a5f',
      'department': 'Visual Effects',
      'gender': 0,
      'id': 1401796,
      'job': 'Visual Effects Supervisor',
      'name': 'John Knoll'},
     {'credit_id': '54959cbbc3a3686ae3004391',
      'department': 'Visual Effects',
      'gender': 2,
      'id': 1401799,
      'job': 'Visual Effects Supervisor',
      'name': 'Eric Saindon'},
     {'credit_id': '54959d06c3a3686ae90043f6',
      'department': 'Visual Effects',
      'gender': 0,
      'id': 1401800,
      'job': 'Visual Effects Supervisor',
      'name': 'Wayne Stables'},
     {'credit_id': '54959d259251416e1e003f11',
      'department': 'Visual Effects',
      'gender': 0,
      'id': 1401801,
      'job': 'Visual Effects Supervisor',
      'name': 'David Stinnett'},
     {'credit_id': '54959db49251413af7002975',
      'department': 'Visual Effects',
      'gender': 0,
      'id': 1401803,
      'job': 'Visual Effects Supervisor',
      'name': 'Guy Williams'},
     {'credit_id': '54959de4c3a3681153002750',
      'department': 'Crew',
      'gender': 0,
      'id': 1401804,
      'job': 'Stunt Coordinator',
      'name': 'Stuart Thorp'},
     {'credit_id': '54959ef2c3a3680fc60027f2',
      'department': 'Lighting',
      'gender': 0,
      'id': 1401805,
      'job': 'Best Boy Electric',
      'name': 'Giles Coburn'},
     {'credit_id': '54959f07c3a3680fc60027f9',
      'department': 'Camera',
      'gender': 2,
      'id': 1401806,
      'job': 'Still Photographer',
      'name': 'Mark Fellman'},
     {'credit_id': '54959f47c3a3681153002774',
      'department': 'Lighting',
      'gender': 0,
      'id': 1401807,
      'job': 'Lighting Technician',
      'name': 'Scott Sprague'},
     {'credit_id': '54959f8cc3a36831b8001df2',
      'department': 'Visual Effects',
      'gender': 0,
      'id': 1401808,
      'job': 'Animation Director',
      'name': 'Jeremy Hollobon'},
     {'credit_id': '54959fa0c3a36831b8001dfb',
      'department': 'Visual Effects',
      'gender': 0,
      'id': 1401809,
      'job': 'Animation Director',
      'name': 'Orlando Meunier'},
     {'credit_id': '54959fb6c3a3686af3003f54',
      'department': 'Visual Effects',
      'gender': 0,
      'id': 1401810,
      'job': 'Animation Director',
      'name': 'Taisuke Tanimura'},
     {'credit_id': '54959fd2c3a36831b8001e02',
      'department': 'Costume & Make-Up',
      'gender': 0,
      'id': 1401812,
      'job': 'Set Costumer',
      'name': 'Lilia Mishel Acevedo'},
     {'credit_id': '54959ff9c3a3686ae300440c',
      'department': 'Costume & Make-Up',
      'gender': 0,
      'id': 1401814,
      'job': 'Set Costumer',
      'name': 'Alejandro M. Hernandez'},
     {'credit_id': '5495a0ddc3a3686ae10046fe',
      'department': 'Editing',
      'gender': 0,
      'id': 1401815,
      'job': 'Digital Intermediate',
      'name': 'Marvin Hall'},
     {'credit_id': '5495a1f7c3a3686ae3004443',
      'department': 'Production',
      'gender': 0,
      'id': 1401816,
      'job': 'Publicist',
      'name': 'Judy Alley'},
     {'credit_id': '5592b29fc3a36869d100002f',
      'department': 'Crew',
      'gender': 0,
      'id': 1418381,
      'job': 'CG Supervisor',
      'name': 'Mike Perry'},
     {'credit_id': '5592b23a9251415df8001081',
      'department': 'Crew',
      'gender': 0,
      'id': 1426854,
      'job': 'CG Supervisor',
      'name': 'Andrew Morley'},
     {'credit_id': '55491e1192514104c40002d8',
      'department': 'Art',
      'gender': 0,
      'id': 1438901,
      'job': 'Conceptual Design',
      'name': 'Seth Engstrom'},
     {'credit_id': '5525d5809251417276002b06',
      'department': 'Crew',
      'gender': 0,
      'id': 1447362,
      'job': 'Visual Effects Art Director',
      'name': 'Eric Oliver'},
     {'credit_id': '554427ca925141586500312a',
      'department': 'Visual Effects',
      'gender': 0,
      'id': 1447503,
      'job': 'Modeling',
      'name': 'Matsune Suzuki'},
     {'credit_id': '551906889251415aab001c88',
      'department': 'Art',
      'gender': 0,
      'id': 1447524,
      'job': 'Art Department Manager',
      'name': 'Paul Tobin'},
     {'credit_id': '5592af8492514152cc0010de',
      'department': 'Costume & Make-Up',
      'gender': 0,
      'id': 1452643,
      'job': 'Hairstylist',
      'name': 'Roxane Griffin'},
     {'credit_id': '553d3c109251415852001318',
      'department': 'Lighting',
      'gender': 0,
      'id': 1453938,
      'job': 'Lighting Artist',
      'name': 'Arun Ram-Mohan'},
     {'credit_id': '5592af4692514152d5001355',
      'department': 'Costume & Make-Up',
      'gender': 0,
      'id': 1457305,
      'job': 'Makeup Artist',
      'name': 'Georgia Lockhart-Adams'},
     {'credit_id': '5592b2eac3a36877470012a5',
      'department': 'Crew',
      'gender': 0,
      'id': 1466035,
      'job': 'CG Supervisor',
      'name': 'Thrain Shadbolt'},
     {'credit_id': '5592b032c3a36877450015f1',
      'department': 'Crew',
      'gender': 0,
      'id': 1483220,
      'job': 'CG Supervisor',
      'name': 'Brad Alexander'},
     {'credit_id': '5592b05592514152d80012f6',
      'department': 'Crew',
      'gender': 0,
      'id': 1483221,
      'job': 'CG Supervisor',
      'name': 'Shadi Almassizadeh'},
     {'credit_id': '5592b090c3a36877570010b5',
      'department': 'Crew',
      'gender': 0,
      'id': 1483222,
      'job': 'CG Supervisor',
      'name': 'Simon Clutterbuck'},
     {'credit_id': '5592b0dbc3a368774b00112c',
      'department': 'Crew',
      'gender': 0,
      'id': 1483223,
      'job': 'CG Supervisor',
      'name': 'Graeme Demmocks'},
     {'credit_id': '5592b0fe92514152db0010c1',
      'department': 'Crew',
      'gender': 0,
      'id': 1483224,
      'job': 'CG Supervisor',
      'name': 'Adrian Fernandes'},
     {'credit_id': '5592b11f9251415df8001059',
      'department': 'Crew',
      'gender': 0,
      'id': 1483225,
      'job': 'CG Supervisor',
      'name': 'Mitch Gates'},
     {'credit_id': '5592b15dc3a3687745001645',
      'department': 'Crew',
      'gender': 0,
      'id': 1483226,
      'job': 'CG Supervisor',
      'name': 'Jerry Kung'},
     {'credit_id': '5592b18e925141645a0004ae',
      'department': 'Crew',
      'gender': 0,
      'id': 1483227,
      'job': 'CG Supervisor',
      'name': 'Andy Lomas'},
     {'credit_id': '5592b1bfc3a368775d0010e7',
      'department': 'Crew',
      'gender': 0,
      'id': 1483228,
      'job': 'CG Supervisor',
      'name': 'Sebastian Marino'},
     {'credit_id': '5592b2049251415df8001078',
      'department': 'Crew',
      'gender': 0,
      'id': 1483229,
      'job': 'CG Supervisor',
      'name': 'Matthias Menz'},
     {'credit_id': '5592b27b92514152d800136a',
      'department': 'Crew',
      'gender': 0,
      'id': 1483230,
      'job': 'CG Supervisor',
      'name': 'Sergei Nevshupov'},
     {'credit_id': '5592b2c3c3a36869e800003c',
      'department': 'Crew',
      'gender': 0,
      'id': 1483231,
      'job': 'CG Supervisor',
      'name': 'Philippe Rebours'},
     {'credit_id': '5592b317c3a36877470012af',
      'department': 'Crew',
      'gender': 0,
      'id': 1483232,
      'job': 'CG Supervisor',
      'name': 'Michael Takarangi'},
     {'credit_id': '5592b345c3a36877470012bb',
      'department': 'Crew',
      'gender': 0,
      'id': 1483233,
      'job': 'CG Supervisor',
      'name': 'David Weitzberg'},
     {'credit_id': '5592b37cc3a368775100113b',
      'department': 'Crew',
      'gender': 0,
      'id': 1483234,
      'job': 'CG Supervisor',
      'name': 'Ben White'},
     {'credit_id': '573c8e2f9251413f5d000094',
      'department': 'Crew',
      'gender': 1,
      'id': 1621932,
      'job': 'Stunts',
      'name': 'Min Windle'}]




```python
# 감독 정보를 추출
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan 
```


```python
df2['director'] = df2['crew'].apply(get_director)
df2['director']
```




    0           James Cameron
    1          Gore Verbinski
    2              Sam Mendes
    3       Christopher Nolan
    4          Andrew Stanton
                  ...        
    4798     Robert Rodriguez
    4799         Edward Burns
    4800          Scott Smith
    4801          Daniel Hsia
    4802     Brian Herzlinger
    Name: director, Length: 4803, dtype: object




```python
df2[df2['director'].isnull()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>keywords</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>popularity</th>
      <th>production_companies</th>
      <th>...</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>cast</th>
      <th>crew</th>
      <th>director</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3661</th>
      <td>0</td>
      <td>[{'id': 18, 'name': 'Drama'}]</td>
      <td>NaN</td>
      <td>19615</td>
      <td>[]</td>
      <td>en</td>
      <td>Flying By</td>
      <td>A real estate developer goes to his 25th high ...</td>
      <td>1.546169</td>
      <td>[]</td>
      <td>...</td>
      <td>95.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>It's about the music</td>
      <td>Flying By</td>
      <td>7.0</td>
      <td>2</td>
      <td>[{'cast_id': 1, 'character': 'George', 'credit...</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3670</th>
      <td>0</td>
      <td>[{'id': 10751, 'name': 'Family'}]</td>
      <td>NaN</td>
      <td>447027</td>
      <td>[]</td>
      <td>en</td>
      <td>Running Forever</td>
      <td>After being estranged since her mother's death...</td>
      <td>0.028756</td>
      <td>[{"name": "New Kingdom Pictures", "id": 41671}]</td>
      <td>...</td>
      <td>88.0</td>
      <td>[]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Running Forever</td>
      <td>0.0</td>
      <td>0</td>
      <td>[]</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3729</th>
      <td>3250000</td>
      <td>[{'id': 18, 'name': 'Drama'}, {'id': 10751, 'n...</td>
      <td>http://www.paathefilm.com/</td>
      <td>26379</td>
      <td>[]</td>
      <td>en</td>
      <td>Paa</td>
      <td>He suffers from a progeria like syndrome. Ment...</td>
      <td>2.126139</td>
      <td>[{"name": "A B Corp", "id": 4502}]</td>
      <td>...</td>
      <td>133.0</td>
      <td>[{"iso_639_1": "hi", "name": "\u0939\u093f\u09...</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Paa</td>
      <td>6.6</td>
      <td>19</td>
      <td>[{'cast_id': 1, 'character': 'Auro', 'credit_i...</td>
      <td>[{'credit_id': '52fe44fec3a368484e042a29', 'de...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3977</th>
      <td>0</td>
      <td>[{'id': 35, 'name': 'Comedy'}, {'id': 18, 'nam...</td>
      <td>NaN</td>
      <td>55831</td>
      <td>[{'id': 10183, 'name': 'independent film'}]</td>
      <td>en</td>
      <td>Boynton Beach Club</td>
      <td>A handful of men and women of a certain age pi...</td>
      <td>0.188870</td>
      <td>[]</td>
      <td>...</td>
      <td>105.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Boynton Beach Club</td>
      <td>6.8</td>
      <td>3</td>
      <td>[{'cast_id': 1, 'character': 'Marilyn', 'credi...</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4068</th>
      <td>0</td>
      <td>[]</td>
      <td>NaN</td>
      <td>371085</td>
      <td>[]</td>
      <td>en</td>
      <td>Sharkskin</td>
      <td>The Post War II story of Manhattan born Mike E...</td>
      <td>0.027801</td>
      <td>[]</td>
      <td>...</td>
      <td>0.0</td>
      <td>[]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Sharkskin</td>
      <td>0.0</td>
      <td>0</td>
      <td>[]</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4105</th>
      <td>2000000</td>
      <td>[]</td>
      <td>NaN</td>
      <td>48382</td>
      <td>[]</td>
      <td>en</td>
      <td>The Book of Mormon Movie, Volume 1: The Journey</td>
      <td>The story of Lehi and his wife Sariah and thei...</td>
      <td>0.031947</td>
      <td>[]</td>
      <td>...</td>
      <td>120.0</td>
      <td>[]</td>
      <td>Released</td>
      <td>2600 years ago, one family began a remarkable ...</td>
      <td>The Book of Mormon Movie, Volume 1: The Journey</td>
      <td>5.0</td>
      <td>2</td>
      <td>[{'cast_id': 1, 'character': 'Sam', 'credit_id...</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4118</th>
      <td>0</td>
      <td>[]</td>
      <td>NaN</td>
      <td>325140</td>
      <td>[]</td>
      <td>en</td>
      <td>Hum To Mohabbat Karega</td>
      <td>Raju, a waiter, is in love with the famous TV ...</td>
      <td>0.001186</td>
      <td>[]</td>
      <td>...</td>
      <td>0.0</td>
      <td>[]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Hum To Mohabbat Karega</td>
      <td>0.0</td>
      <td>0</td>
      <td>[]</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4123</th>
      <td>7000000</td>
      <td>[{'id': 16, 'name': 'Animation'}, {'id': 10751...</td>
      <td>http://www.roadsideromeo.com/</td>
      <td>20653</td>
      <td>[]</td>
      <td>en</td>
      <td>Roadside Romeo</td>
      <td>This is the story of Romeo. A dude who was liv...</td>
      <td>0.253595</td>
      <td>[{"name": "Walt Disney Pictures", "id": 2}, {"...</td>
      <td>...</td>
      <td>93.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}, {"iso...</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Roadside Romeo</td>
      <td>6.7</td>
      <td>3</td>
      <td>[{'cast_id': 1, 'character': 'Romeo', 'credit_...</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4247</th>
      <td>1</td>
      <td>[{'id': 10749, 'name': 'Romance'}, {'id': 35, ...</td>
      <td>NaN</td>
      <td>361505</td>
      <td>[]</td>
      <td>en</td>
      <td>Me You and Five Bucks</td>
      <td>A womanizing yet lovable loser, Charlie, a wai...</td>
      <td>0.094105</td>
      <td>[]</td>
      <td>...</td>
      <td>90.0</td>
      <td>[]</td>
      <td>Released</td>
      <td>A story about second, second chances</td>
      <td>Me You and Five Bucks</td>
      <td>10.0</td>
      <td>2</td>
      <td>[]</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4305</th>
      <td>0</td>
      <td>[{'id': 35, 'name': 'Comedy'}, {'id': 10402, '...</td>
      <td>NaN</td>
      <td>114065</td>
      <td>[]</td>
      <td>en</td>
      <td>Down &amp; Out With The Dolls</td>
      <td>The raunchy, spunky tale of the rise and fall ...</td>
      <td>0.002386</td>
      <td>[]</td>
      <td>...</td>
      <td>88.0</td>
      <td>[]</td>
      <td>Released</td>
      <td>Ain't Rock 'N' Roll a bitch.</td>
      <td>Down &amp; Out With The Dolls</td>
      <td>0.0</td>
      <td>0</td>
      <td>[]</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4314</th>
      <td>1200000</td>
      <td>[]</td>
      <td>NaN</td>
      <td>137955</td>
      <td>[]</td>
      <td>en</td>
      <td>Crowsnest</td>
      <td>In late summer of 2011, five young friends on ...</td>
      <td>0.057564</td>
      <td>[]</td>
      <td>...</td>
      <td>84.0</td>
      <td>[]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Crowsnest</td>
      <td>4.8</td>
      <td>12</td>
      <td>[]</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4322</th>
      <td>0</td>
      <td>[{'id': 99, 'name': 'Documentary'}]</td>
      <td>NaN</td>
      <td>102840</td>
      <td>[]</td>
      <td>en</td>
      <td>Sex With Strangers</td>
      <td>For some married couples, sex is an obsession ...</td>
      <td>0.014406</td>
      <td>[]</td>
      <td>...</td>
      <td>0.0</td>
      <td>[]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Sex With Strangers</td>
      <td>5.0</td>
      <td>1</td>
      <td>[]</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4374</th>
      <td>0</td>
      <td>[{'id': 35, 'name': 'Comedy'}]</td>
      <td>NaN</td>
      <td>47686</td>
      <td>[{'id': 10183, 'name': 'independent film'}]</td>
      <td>en</td>
      <td>Dream with the Fishes</td>
      <td>Terry is a suicidal voyeur who treats a dying ...</td>
      <td>0.948316</td>
      <td>[]</td>
      <td>...</td>
      <td>97.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>An oddball odyssey about voyeurism, LSD and nu...</td>
      <td>Dream with the Fishes</td>
      <td>7.7</td>
      <td>10</td>
      <td>[{'cast_id': 1, 'character': 'Terry', 'credit_...</td>
      <td>[{'credit_id': '555e51909251417e5f000b42', 'de...</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4401</th>
      <td>0</td>
      <td>[{'id': 28, 'name': 'Action'}, {'id': 35, 'nam...</td>
      <td>NaN</td>
      <td>43630</td>
      <td>[]</td>
      <td>en</td>
      <td>The Helix... Loaded</td>
      <td></td>
      <td>0.020600</td>
      <td>[]</td>
      <td>...</td>
      <td>97.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Rumored</td>
      <td>NaN</td>
      <td>The Helix... Loaded</td>
      <td>4.8</td>
      <td>2</td>
      <td>[]</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4405</th>
      <td>0</td>
      <td>[{'id': 10751, 'name': 'Family'}, {'id': 35, '...</td>
      <td>https://www.epicbuzz.net/movies/karachi-se-lahore</td>
      <td>357441</td>
      <td>[]</td>
      <td>en</td>
      <td>Karachi se Lahore</td>
      <td>A road trip from Karachi to Lahore where 5 fri...</td>
      <td>0.060003</td>
      <td>[]</td>
      <td>...</td>
      <td>0.0</td>
      <td>[{"iso_639_1": "ur", "name": "\u0627\u0631\u06...</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Karachi se Lahore</td>
      <td>8.0</td>
      <td>1</td>
      <td>[{'cast_id': 0, 'character': '', 'credit_id': ...</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4458</th>
      <td>0</td>
      <td>[]</td>
      <td>NaN</td>
      <td>279759</td>
      <td>[]</td>
      <td>en</td>
      <td>Harrison Montgomery</td>
      <td>Film from Daniel Davila</td>
      <td>0.006943</td>
      <td>[]</td>
      <td>...</td>
      <td>0.0</td>
      <td>[]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Harrison Montgomery</td>
      <td>0.0</td>
      <td>0</td>
      <td>[]</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4504</th>
      <td>0</td>
      <td>[]</td>
      <td>NaN</td>
      <td>331493</td>
      <td>[]</td>
      <td>en</td>
      <td>Light from the Darkroom</td>
      <td>Light in the Darkroom is the story of two best...</td>
      <td>0.012942</td>
      <td>[]</td>
      <td>...</td>
      <td>0.0</td>
      <td>[]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Light from the Darkroom</td>
      <td>0.0</td>
      <td>0</td>
      <td>[]</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4553</th>
      <td>0</td>
      <td>[]</td>
      <td>NaN</td>
      <td>380097</td>
      <td>[]</td>
      <td>en</td>
      <td>America Is Still the Place</td>
      <td>1971 post civil rights San Francisco seemed li...</td>
      <td>0.000000</td>
      <td>[]</td>
      <td>...</td>
      <td>0.0</td>
      <td>[]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>America Is Still the Place</td>
      <td>0.0</td>
      <td>0</td>
      <td>[]</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4562</th>
      <td>500000</td>
      <td>[]</td>
      <td>NaN</td>
      <td>297100</td>
      <td>[]</td>
      <td>en</td>
      <td>The Little Ponderosa Zoo</td>
      <td>The Little Ponderosa Zoo is preparing for thei...</td>
      <td>0.073079</td>
      <td>[]</td>
      <td>...</td>
      <td>84.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>The Little Ponderosa Zoo</td>
      <td>2.0</td>
      <td>1</td>
      <td>[]</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4566</th>
      <td>0</td>
      <td>[]</td>
      <td>NaN</td>
      <td>325579</td>
      <td>[]</td>
      <td>en</td>
      <td>Diamond Ruff</td>
      <td>Action - Orphan, con artist, crime boss and mi...</td>
      <td>0.165257</td>
      <td>[]</td>
      <td>...</td>
      <td>0.0</td>
      <td>[]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Diamond Ruff</td>
      <td>2.4</td>
      <td>4</td>
      <td>[]</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4571</th>
      <td>0</td>
      <td>[]</td>
      <td>NaN</td>
      <td>328307</td>
      <td>[]</td>
      <td>en</td>
      <td>Rise of the Entrepreneur: The Search for a Bet...</td>
      <td>The world is changing faster than ever. Techno...</td>
      <td>0.052942</td>
      <td>[]</td>
      <td>...</td>
      <td>0.0</td>
      <td>[]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Rise of the Entrepreneur: The Search for a Bet...</td>
      <td>8.0</td>
      <td>1</td>
      <td>[]</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4583</th>
      <td>0</td>
      <td>[{'id': 99, 'name': 'Documentary'}]</td>
      <td>http://www.iwantyourmoney.net/</td>
      <td>47546</td>
      <td>[]</td>
      <td>en</td>
      <td>I Want Your Money</td>
      <td>Two versions of the American dream now stand i...</td>
      <td>0.084344</td>
      <td>[]</td>
      <td>...</td>
      <td>92.0</td>
      <td>[]</td>
      <td>Released</td>
      <td>The film contrasts two views of role that the ...</td>
      <td>I Want Your Money</td>
      <td>3.8</td>
      <td>5</td>
      <td>[]</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4589</th>
      <td>0</td>
      <td>[{'id': 18, 'name': 'Drama'}, {'id': 9648, 'na...</td>
      <td>NaN</td>
      <td>43743</td>
      <td>[{'id': 10183, 'name': 'independent film'}]</td>
      <td>en</td>
      <td>Fabled</td>
      <td>Joseph just broke up with his girlfriend and i...</td>
      <td>0.003352</td>
      <td>[]</td>
      <td>...</td>
      <td>84.0</td>
      <td>[]</td>
      <td>Released</td>
      <td>There once was a wolf named Lupold...</td>
      <td>Fabled</td>
      <td>0.0</td>
      <td>0</td>
      <td>[]</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4633</th>
      <td>0</td>
      <td>[]</td>
      <td>NaN</td>
      <td>300327</td>
      <td>[]</td>
      <td>en</td>
      <td>Death Calls</td>
      <td>An action-packed love story on the Mexican bor...</td>
      <td>0.005883</td>
      <td>[]</td>
      <td>...</td>
      <td>0.0</td>
      <td>[]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Death Calls</td>
      <td>0.0</td>
      <td>0</td>
      <td>[]</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4638</th>
      <td>300000</td>
      <td>[{'id': 18, 'name': 'Drama'}, {'id': 28, 'name...</td>
      <td>NaN</td>
      <td>378237</td>
      <td>[]</td>
      <td>en</td>
      <td>Amidst the Devil's Wings</td>
      <td>Prequel to "5th of a Degree."</td>
      <td>0.018087</td>
      <td>[{"name": "Daniel Columbie Films &amp; Productions...</td>
      <td>...</td>
      <td>90.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>Prequel to "5th of a Degree."</td>
      <td>Amidst the Devil's Wings</td>
      <td>0.0</td>
      <td>0</td>
      <td>[]</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4644</th>
      <td>0</td>
      <td>[{'id': 27, 'name': 'Horror'}]</td>
      <td>NaN</td>
      <td>325123</td>
      <td>[]</td>
      <td>en</td>
      <td>Teeth and Blood</td>
      <td>A beautiful diva is murdered on the set of hor...</td>
      <td>0.055325</td>
      <td>[]</td>
      <td>...</td>
      <td>96.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Teeth and Blood</td>
      <td>3.0</td>
      <td>1</td>
      <td>[{'cast_id': 0, 'character': 'Vincent Augustin...</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4657</th>
      <td>0</td>
      <td>[]</td>
      <td>NaN</td>
      <td>320435</td>
      <td>[]</td>
      <td>en</td>
      <td>UnDivided</td>
      <td>UnDivided documents the true story of how a su...</td>
      <td>0.010607</td>
      <td>[]</td>
      <td>...</td>
      <td>0.0</td>
      <td>[]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>UnDivided</td>
      <td>0.0</td>
      <td>0</td>
      <td>[]</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4662</th>
      <td>0</td>
      <td>[{'id': 35, 'name': 'Comedy'}]</td>
      <td>NaN</td>
      <td>40963</td>
      <td>[{'id': 10183, 'name': 'independent film'}]</td>
      <td>en</td>
      <td>Little Big Top</td>
      <td>An aging out of work clown returns to his smal...</td>
      <td>0.092100</td>
      <td>[{"name": "Fly High Films", "id": 24248}]</td>
      <td>...</td>
      <td>0.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Rumored</td>
      <td>NaN</td>
      <td>Little Big Top</td>
      <td>10.0</td>
      <td>1</td>
      <td>[{'cast_id': 0, 'character': 'Seymour', 'credi...</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4674</th>
      <td>0</td>
      <td>[]</td>
      <td>NaN</td>
      <td>194588</td>
      <td>[]</td>
      <td>en</td>
      <td>Short Cut to Nirvana: Kumbh Mela</td>
      <td>Every 12 years over 70 million pilgrims gather...</td>
      <td>0.004998</td>
      <td>[]</td>
      <td>...</td>
      <td>85.0</td>
      <td>[]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Short Cut to Nirvana: Kumbh Mela</td>
      <td>0.0</td>
      <td>0</td>
      <td>[]</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4716</th>
      <td>0</td>
      <td>[]</td>
      <td>NaN</td>
      <td>38786</td>
      <td>[]</td>
      <td>en</td>
      <td>The Blood of My Brother: A Story of Death in Iraq</td>
      <td>THE BLOOD OF MY BROTHER goes behind the scenes...</td>
      <td>0.005256</td>
      <td>[]</td>
      <td>...</td>
      <td>90.0</td>
      <td>[]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>The Blood of My Brother: A Story of Death in Iraq</td>
      <td>0.0</td>
      <td>0</td>
      <td>[]</td>
      <td>[]</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>30 rows × 23 columns</p>
</div>



출연배우들(cast) 정보


```python
df2.loc[0, 'cast'] #cast 데이터 가져오기
```




    [{'cast_id': 242,
      'character': 'Jake Sully',
      'credit_id': '5602a8a7c3a3685532001c9a',
      'gender': 2,
      'id': 65731,
      'name': 'Sam Worthington',
      'order': 0},
     {'cast_id': 3,
      'character': 'Neytiri',
      'credit_id': '52fe48009251416c750ac9cb',
      'gender': 1,
      'id': 8691,
      'name': 'Zoe Saldana',
      'order': 1},
     {'cast_id': 25,
      'character': 'Dr. Grace Augustine',
      'credit_id': '52fe48009251416c750aca39',
      'gender': 1,
      'id': 10205,
      'name': 'Sigourney Weaver',
      'order': 2},
     {'cast_id': 4,
      'character': 'Col. Quaritch',
      'credit_id': '52fe48009251416c750ac9cf',
      'gender': 2,
      'id': 32747,
      'name': 'Stephen Lang',
      'order': 3},
     {'cast_id': 5,
      'character': 'Trudy Chacon',
      'credit_id': '52fe48009251416c750ac9d3',
      'gender': 1,
      'id': 17647,
      'name': 'Michelle Rodriguez',
      'order': 4},
     {'cast_id': 8,
      'character': 'Selfridge',
      'credit_id': '52fe48009251416c750ac9e1',
      'gender': 2,
      'id': 1771,
      'name': 'Giovanni Ribisi',
      'order': 5},
     {'cast_id': 7,
      'character': 'Norm Spellman',
      'credit_id': '52fe48009251416c750ac9dd',
      'gender': 2,
      'id': 59231,
      'name': 'Joel David Moore',
      'order': 6},
     {'cast_id': 9,
      'character': 'Moat',
      'credit_id': '52fe48009251416c750ac9e5',
      'gender': 1,
      'id': 30485,
      'name': 'CCH Pounder',
      'order': 7},
     {'cast_id': 11,
      'character': 'Eytukan',
      'credit_id': '52fe48009251416c750ac9ed',
      'gender': 2,
      'id': 15853,
      'name': 'Wes Studi',
      'order': 8},
     {'cast_id': 10,
      'character': "Tsu'Tey",
      'credit_id': '52fe48009251416c750ac9e9',
      'gender': 2,
      'id': 10964,
      'name': 'Laz Alonso',
      'order': 9},
     {'cast_id': 12,
      'character': 'Dr. Max Patel',
      'credit_id': '52fe48009251416c750ac9f1',
      'gender': 2,
      'id': 95697,
      'name': 'Dileep Rao',
      'order': 10},
     {'cast_id': 13,
      'character': 'Lyle Wainfleet',
      'credit_id': '52fe48009251416c750ac9f5',
      'gender': 2,
      'id': 98215,
      'name': 'Matt Gerald',
      'order': 11},
     {'cast_id': 32,
      'character': 'Private Fike',
      'credit_id': '52fe48009251416c750aca5b',
      'gender': 2,
      'id': 154153,
      'name': 'Sean Anthony Moran',
      'order': 12},
     {'cast_id': 33,
      'character': 'Cryo Vault Med Tech',
      'credit_id': '52fe48009251416c750aca5f',
      'gender': 2,
      'id': 397312,
      'name': 'Jason Whyte',
      'order': 13},
     {'cast_id': 34,
      'character': 'Venture Star Crew Chief',
      'credit_id': '52fe48009251416c750aca63',
      'gender': 2,
      'id': 42317,
      'name': 'Scott Lawrence',
      'order': 14},
     {'cast_id': 35,
      'character': 'Lock Up Trooper',
      'credit_id': '52fe48009251416c750aca67',
      'gender': 2,
      'id': 986734,
      'name': 'Kelly Kilgour',
      'order': 15},
     {'cast_id': 36,
      'character': 'Shuttle Pilot',
      'credit_id': '52fe48009251416c750aca6b',
      'gender': 0,
      'id': 1207227,
      'name': 'James Patrick Pitt',
      'order': 16},
     {'cast_id': 37,
      'character': 'Shuttle Co-Pilot',
      'credit_id': '52fe48009251416c750aca6f',
      'gender': 0,
      'id': 1180936,
      'name': 'Sean Patrick Murphy',
      'order': 17},
     {'cast_id': 38,
      'character': 'Shuttle Crew Chief',
      'credit_id': '52fe48009251416c750aca73',
      'gender': 2,
      'id': 1019578,
      'name': 'Peter Dillon',
      'order': 18},
     {'cast_id': 39,
      'character': 'Tractor Operator / Troupe',
      'credit_id': '52fe48009251416c750aca77',
      'gender': 0,
      'id': 91443,
      'name': 'Kevin Dorman',
      'order': 19},
     {'cast_id': 40,
      'character': 'Dragon Gunship Pilot',
      'credit_id': '52fe48009251416c750aca7b',
      'gender': 2,
      'id': 173391,
      'name': 'Kelson Henderson',
      'order': 20},
     {'cast_id': 41,
      'character': 'Dragon Gunship Gunner',
      'credit_id': '52fe48009251416c750aca7f',
      'gender': 0,
      'id': 1207236,
      'name': 'David Van Horn',
      'order': 21},
     {'cast_id': 42,
      'character': 'Dragon Gunship Navigator',
      'credit_id': '52fe48009251416c750aca83',
      'gender': 0,
      'id': 215913,
      'name': 'Jacob Tomuri',
      'order': 22},
     {'cast_id': 43,
      'character': 'Suit #1',
      'credit_id': '52fe48009251416c750aca87',
      'gender': 0,
      'id': 143206,
      'name': 'Michael Blain-Rozgay',
      'order': 23},
     {'cast_id': 44,
      'character': 'Suit #2',
      'credit_id': '52fe48009251416c750aca8b',
      'gender': 2,
      'id': 169676,
      'name': 'Jon Curry',
      'order': 24},
     {'cast_id': 46,
      'character': 'Ambient Room Tech',
      'credit_id': '52fe48009251416c750aca8f',
      'gender': 0,
      'id': 1048610,
      'name': 'Luke Hawker',
      'order': 25},
     {'cast_id': 47,
      'character': 'Ambient Room Tech / Troupe',
      'credit_id': '52fe48009251416c750aca93',
      'gender': 0,
      'id': 42288,
      'name': 'Woody Schultz',
      'order': 26},
     {'cast_id': 48,
      'character': 'Horse Clan Leader',
      'credit_id': '52fe48009251416c750aca97',
      'gender': 2,
      'id': 68278,
      'name': 'Peter Mensah',
      'order': 27},
     {'cast_id': 49,
      'character': 'Link Room Tech',
      'credit_id': '52fe48009251416c750aca9b',
      'gender': 0,
      'id': 1207247,
      'name': 'Sonia Yee',
      'order': 28},
     {'cast_id': 50,
      'character': 'Basketball Avatar / Troupe',
      'credit_id': '52fe48009251416c750aca9f',
      'gender': 1,
      'id': 1207248,
      'name': 'Jahnel Curfman',
      'order': 29},
     {'cast_id': 51,
      'character': 'Basketball Avatar',
      'credit_id': '52fe48009251416c750acaa3',
      'gender': 0,
      'id': 89714,
      'name': 'Ilram Choi',
      'order': 30},
     {'cast_id': 52,
      'character': "Na'vi Child",
      'credit_id': '52fe48009251416c750acaa7',
      'gender': 0,
      'id': 1207249,
      'name': 'Kyla Warren',
      'order': 31},
     {'cast_id': 53,
      'character': 'Troupe',
      'credit_id': '52fe48009251416c750acaab',
      'gender': 0,
      'id': 1207250,
      'name': 'Lisa Roumain',
      'order': 32},
     {'cast_id': 54,
      'character': 'Troupe',
      'credit_id': '52fe48009251416c750acaaf',
      'gender': 1,
      'id': 83105,
      'name': 'Debra Wilson',
      'order': 33},
     {'cast_id': 57,
      'character': 'Troupe',
      'credit_id': '52fe48009251416c750acabb',
      'gender': 0,
      'id': 1207253,
      'name': 'Chris Mala',
      'order': 34},
     {'cast_id': 55,
      'character': 'Troupe',
      'credit_id': '52fe48009251416c750acab3',
      'gender': 0,
      'id': 1207251,
      'name': 'Taylor Kibby',
      'order': 35},
     {'cast_id': 56,
      'character': 'Troupe',
      'credit_id': '52fe48009251416c750acab7',
      'gender': 0,
      'id': 1207252,
      'name': 'Jodie Landau',
      'order': 36},
     {'cast_id': 58,
      'character': 'Troupe',
      'credit_id': '52fe48009251416c750acabf',
      'gender': 0,
      'id': 1207254,
      'name': 'Julie Lamm',
      'order': 37},
     {'cast_id': 59,
      'character': 'Troupe',
      'credit_id': '52fe48009251416c750acac3',
      'gender': 0,
      'id': 1207257,
      'name': 'Cullen B. Madden',
      'order': 38},
     {'cast_id': 60,
      'character': 'Troupe',
      'credit_id': '52fe48009251416c750acac7',
      'gender': 0,
      'id': 1207259,
      'name': 'Joseph Brady Madden',
      'order': 39},
     {'cast_id': 61,
      'character': 'Troupe',
      'credit_id': '52fe48009251416c750acacb',
      'gender': 0,
      'id': 1207262,
      'name': 'Frankie Torres',
      'order': 40},
     {'cast_id': 62,
      'character': 'Troupe',
      'credit_id': '52fe48009251416c750acacf',
      'gender': 1,
      'id': 1158600,
      'name': 'Austin Wilson',
      'order': 41},
     {'cast_id': 63,
      'character': 'Troupe',
      'credit_id': '52fe48019251416c750acad3',
      'gender': 1,
      'id': 983705,
      'name': 'Sara Wilson',
      'order': 42},
     {'cast_id': 64,
      'character': 'Troupe',
      'credit_id': '52fe48019251416c750acad7',
      'gender': 0,
      'id': 1207263,
      'name': 'Tamica Washington-Miller',
      'order': 43},
     {'cast_id': 65,
      'character': 'Op Center Staff',
      'credit_id': '52fe48019251416c750acadb',
      'gender': 1,
      'id': 1145098,
      'name': 'Lucy Briant',
      'order': 44},
     {'cast_id': 66,
      'character': 'Op Center Staff',
      'credit_id': '52fe48019251416c750acadf',
      'gender': 2,
      'id': 33305,
      'name': 'Nathan Meister',
      'order': 45},
     {'cast_id': 67,
      'character': 'Op Center Staff',
      'credit_id': '52fe48019251416c750acae3',
      'gender': 0,
      'id': 1207264,
      'name': 'Gerry Blair',
      'order': 46},
     {'cast_id': 68,
      'character': 'Op Center Staff',
      'credit_id': '52fe48019251416c750acae7',
      'gender': 2,
      'id': 33311,
      'name': 'Matthew Chamberlain',
      'order': 47},
     {'cast_id': 69,
      'character': 'Op Center Staff',
      'credit_id': '52fe48019251416c750acaeb',
      'gender': 0,
      'id': 1207265,
      'name': 'Paul Yates',
      'order': 48},
     {'cast_id': 70,
      'character': 'Op Center Duty Officer',
      'credit_id': '52fe48019251416c750acaef',
      'gender': 0,
      'id': 1207266,
      'name': 'Wray Wilson',
      'order': 49},
     {'cast_id': 71,
      'character': 'Op Center Staff',
      'credit_id': '52fe48019251416c750acaf3',
      'gender': 2,
      'id': 54492,
      'name': 'James Gaylyn',
      'order': 50},
     {'cast_id': 72,
      'character': 'Dancer',
      'credit_id': '52fe48019251416c750acaf7',
      'gender': 0,
      'id': 1207267,
      'name': 'Melvin Leno Clark III',
      'order': 51},
     {'cast_id': 73,
      'character': 'Dancer',
      'credit_id': '52fe48019251416c750acafb',
      'gender': 0,
      'id': 1207268,
      'name': 'Carvon Futrell',
      'order': 52},
     {'cast_id': 74,
      'character': 'Dancer',
      'credit_id': '52fe48019251416c750acaff',
      'gender': 0,
      'id': 1207269,
      'name': 'Brandon Jelkes',
      'order': 53},
     {'cast_id': 75,
      'character': 'Dancer',
      'credit_id': '52fe48019251416c750acb03',
      'gender': 0,
      'id': 1207270,
      'name': 'Micah Moch',
      'order': 54},
     {'cast_id': 76,
      'character': 'Dancer',
      'credit_id': '52fe48019251416c750acb07',
      'gender': 0,
      'id': 1207271,
      'name': 'Hanniyah Muhammad',
      'order': 55},
     {'cast_id': 77,
      'character': 'Dancer',
      'credit_id': '52fe48019251416c750acb0b',
      'gender': 0,
      'id': 1207272,
      'name': 'Christopher Nolen',
      'order': 56},
     {'cast_id': 78,
      'character': 'Dancer',
      'credit_id': '52fe48019251416c750acb0f',
      'gender': 0,
      'id': 1207273,
      'name': 'Christa Oliver',
      'order': 57},
     {'cast_id': 79,
      'character': 'Dancer',
      'credit_id': '52fe48019251416c750acb13',
      'gender': 0,
      'id': 1207274,
      'name': 'April Marie Thomas',
      'order': 58},
     {'cast_id': 80,
      'character': 'Dancer',
      'credit_id': '52fe48019251416c750acb17',
      'gender': 0,
      'id': 1207275,
      'name': 'Bravita A. Threatt',
      'order': 59},
     {'cast_id': 81,
      'character': 'Mining Chief (uncredited)',
      'credit_id': '52fe48019251416c750acb1b',
      'gender': 0,
      'id': 1207276,
      'name': 'Colin Bleasdale',
      'order': 60},
     {'cast_id': 82,
      'character': 'Veteran Miner (uncredited)',
      'credit_id': '52fe48019251416c750acb1f',
      'gender': 0,
      'id': 107969,
      'name': 'Mike Bodnar',
      'order': 61},
     {'cast_id': 83,
      'character': 'Richard (uncredited)',
      'credit_id': '52fe48019251416c750acb23',
      'gender': 0,
      'id': 1207278,
      'name': 'Matt Clayton',
      'order': 62},
     {'cast_id': 84,
      'character': "Nav'i (uncredited)",
      'credit_id': '52fe48019251416c750acb27',
      'gender': 1,
      'id': 147898,
      'name': 'Nicole Dionne',
      'order': 63},
     {'cast_id': 85,
      'character': 'Trooper (uncredited)',
      'credit_id': '52fe48019251416c750acb2b',
      'gender': 0,
      'id': 1207280,
      'name': 'Jamie Harrison',
      'order': 64},
     {'cast_id': 86,
      'character': 'Trooper (uncredited)',
      'credit_id': '52fe48019251416c750acb2f',
      'gender': 0,
      'id': 1207281,
      'name': 'Allan Henry',
      'order': 65},
     {'cast_id': 87,
      'character': 'Ground Technician (uncredited)',
      'credit_id': '52fe48019251416c750acb33',
      'gender': 2,
      'id': 1207282,
      'name': 'Anthony Ingruber',
      'order': 66},
     {'cast_id': 88,
      'character': 'Flight Crew Mechanic (uncredited)',
      'credit_id': '52fe48019251416c750acb37',
      'gender': 0,
      'id': 1207283,
      'name': 'Ashley Jeffery',
      'order': 67},
     {'cast_id': 14,
      'character': 'Samson Pilot',
      'credit_id': '52fe48009251416c750ac9f9',
      'gender': 0,
      'id': 98216,
      'name': 'Dean Knowsley',
      'order': 68},
     {'cast_id': 89,
      'character': 'Trooper (uncredited)',
      'credit_id': '52fe48019251416c750acb3b',
      'gender': 0,
      'id': 1201399,
      'name': 'Joseph Mika-Hunt',
      'order': 69},
     {'cast_id': 90,
      'character': 'Banshee (uncredited)',
      'credit_id': '52fe48019251416c750acb3f',
      'gender': 0,
      'id': 236696,
      'name': 'Terry Notary',
      'order': 70},
     {'cast_id': 91,
      'character': 'Soldier (uncredited)',
      'credit_id': '52fe48019251416c750acb43',
      'gender': 0,
      'id': 1207287,
      'name': 'Kai Pantano',
      'order': 71},
     {'cast_id': 92,
      'character': 'Blast Technician (uncredited)',
      'credit_id': '52fe48019251416c750acb47',
      'gender': 0,
      'id': 1207288,
      'name': 'Logan Pithyou',
      'order': 72},
     {'cast_id': 93,
      'character': 'Vindum Raah (uncredited)',
      'credit_id': '52fe48019251416c750acb4b',
      'gender': 0,
      'id': 1207289,
      'name': 'Stuart Pollock',
      'order': 73},
     {'cast_id': 94,
      'character': 'Hero (uncredited)',
      'credit_id': '52fe48019251416c750acb4f',
      'gender': 0,
      'id': 584868,
      'name': 'Raja',
      'order': 74},
     {'cast_id': 95,
      'character': 'Ops Centreworker (uncredited)',
      'credit_id': '52fe48019251416c750acb53',
      'gender': 0,
      'id': 1207290,
      'name': 'Gareth Ruck',
      'order': 75},
     {'cast_id': 96,
      'character': 'Engineer (uncredited)',
      'credit_id': '52fe48019251416c750acb57',
      'gender': 0,
      'id': 1062463,
      'name': 'Rhian Sheehan',
      'order': 76},
     {'cast_id': 97,
      'character': "Col. Quaritch's Mech Suit (uncredited)",
      'credit_id': '52fe48019251416c750acb5b',
      'gender': 0,
      'id': 60656,
      'name': 'T. J. Storm',
      'order': 77},
     {'cast_id': 98,
      'character': 'Female Marine (uncredited)',
      'credit_id': '52fe48019251416c750acb5f',
      'gender': 0,
      'id': 1207291,
      'name': 'Jodie Taylor',
      'order': 78},
     {'cast_id': 99,
      'character': 'Ikran Clan Leader (uncredited)',
      'credit_id': '52fe48019251416c750acb63',
      'gender': 1,
      'id': 1186027,
      'name': 'Alicia Vela-Bailey',
      'order': 79},
     {'cast_id': 100,
      'character': 'Geologist (uncredited)',
      'credit_id': '52fe48019251416c750acb67',
      'gender': 0,
      'id': 1207292,
      'name': 'Richard Whiteside',
      'order': 80},
     {'cast_id': 101,
      'character': "Na'vi (uncredited)",
      'credit_id': '52fe48019251416c750acb6b',
      'gender': 0,
      'id': 103259,
      'name': 'Nikie Zambo',
      'order': 81},
     {'cast_id': 102,
      'character': 'Ambient Room Tech / Troupe',
      'credit_id': '52fe48019251416c750acb6f',
      'gender': 1,
      'id': 42286,
      'name': 'Julene Renee',
      'order': 82}]




```python
df2.loc[0, 'genres'] #장르 데이터 가져오기
```




    [{'id': 28, 'name': 'Action'},
     {'id': 12, 'name': 'Adventure'},
     {'id': 14, 'name': 'Fantasy'},
     {'id': 878, 'name': 'Science Fiction'}]




```python
df2.loc[0, 'keywords'] #키워드 데이터 가져오기
```




    [{'id': 1463, 'name': 'culture clash'},
     {'id': 2964, 'name': 'future'},
     {'id': 3386, 'name': 'space war'},
     {'id': 3388, 'name': 'space colony'},
     {'id': 3679, 'name': 'society'},
     {'id': 3801, 'name': 'space travel'},
     {'id': 9685, 'name': 'futuristic'},
     {'id': 9840, 'name': 'romance'},
     {'id': 9882, 'name': 'space'},
     {'id': 9951, 'name': 'alien'},
     {'id': 10148, 'name': 'tribe'},
     {'id': 10158, 'name': 'alien planet'},
     {'id': 10987, 'name': 'cgi'},
     {'id': 11399, 'name': 'marine'},
     {'id': 13065, 'name': 'soldier'},
     {'id': 14643, 'name': 'battle'},
     {'id': 14720, 'name': 'love affair'},
     {'id': 165431, 'name': 'anti war'},
     {'id': 193554, 'name': 'power relations'},
     {'id': 206690, 'name': 'mind and soul'},
     {'id': 209714, 'name': '3d'}]




```python
# 처음 3개의 데이터 중에서 name 에 해당하는 value 만 추출
def get_list(x):
    if isinstance(x, list): #리스트,튜플인지 x를 확인하는 함수
        names = [i['name'] for i in x]
        if len(names) > 3: ###3명을 말하는건지 이름앞에 3문자를 말하는건지??###
            names = names[:3]
        return names
    return []
```


```python
features = ['cast', 'keywords', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(get_list)
```


```python
df2[['title', 'cast', 'director', 'keywords', 'genres']].head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>cast</th>
      <th>director</th>
      <th>keywords</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Avatar</td>
      <td>[Sam Worthington, Zoe Saldana, Sigourney Weaver]</td>
      <td>James Cameron</td>
      <td>[culture clash, future, space war]</td>
      <td>[Action, Adventure, Fantasy]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>[Johnny Depp, Orlando Bloom, Keira Knightley]</td>
      <td>Gore Verbinski</td>
      <td>[ocean, drug abuse, exotic island]</td>
      <td>[Adventure, Fantasy, Action]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Spectre</td>
      <td>[Daniel Craig, Christoph Waltz, Léa Seydoux]</td>
      <td>Sam Mendes</td>
      <td>[spy, based on novel, secret agent]</td>
      <td>[Action, Adventure, Crime]</td>
    </tr>
  </tbody>
</table>
</div>



데이터를 소문자로 바꾸고, 빈데이터를 없에주기


```python
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(' ', '')) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(' ', ''))
        else:
            return ''
```


```python
features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    df2[feature] = df2[feature].apply(clean_data)
```


```python
df2[['title', 'cast', 'director', 'keywords', 'genres']].head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>cast</th>
      <th>director</th>
      <th>keywords</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Avatar</td>
      <td>[samworthington, zoesaldana, sigourneyweaver]</td>
      <td>jamescameron</td>
      <td>[cultureclash, future, spacewar]</td>
      <td>[action, adventure, fantasy]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>[johnnydepp, orlandobloom, keiraknightley]</td>
      <td>goreverbinski</td>
      <td>[ocean, drugabuse, exoticisland]</td>
      <td>[adventure, fantasy, action]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Spectre</td>
      <td>[danielcraig, christophwaltz, léaseydoux]</td>
      <td>sammendes</td>
      <td>[spy, basedonnovel, secretagent]</td>
      <td>[action, adventure, crime]</td>
    </tr>
  </tbody>
</table>
</div>



 콤마대신 띄어쓰기로 바꾸는 soup컬럼 만들기


```python
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
df2['soup'] = df2.apply(create_soup, axis=1) ###create_soup함수에 x값이 없어도 되는지?###
df2['soup']
```




    0       cultureclash future spacewar samworthington zo...
    1       ocean drugabuse exoticisland johnnydepp orland...
    2       spy basedonnovel secretagent danielcraig chris...
    3       dccomics crimefighter terrorist christianbale ...
    4       basedonnovel mars medallion taylorkitsch lynnc...
                                  ...                        
    4798    unitedstates–mexicobarrier legs arms carlosgal...
    4799     edwardburns kerrybishé marshadietlein edwardb...
    4800    date loveatfirstsight narration ericmabius kri...
    4801       danielhenney elizacoupe billpaxton danielhsia 
    4802    obsession camcorder crush drewbarrymore brianh...
    Name: soup, Length: 4803, dtype: object



문자의 수를 확인하기


```python
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df2['soup'])
count_matrix
```




    <4803x11520 sparse matrix of type '<class 'numpy.int64'>'
    	with 42935 stored elements in Compressed Sparse Row format>



새로운 코사인 유사도 만들기


```python
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
cosine_sim2
```




    array([[1. , 0.3, 0.2, ..., 0. , 0. , 0. ],
           [0.3, 1. , 0.2, ..., 0. , 0. , 0. ],
           [0.2, 0.2, 1. , ..., 0. , 0. , 0. ],
           ...,
           [0. , 0. , 0. , ..., 1. , 0. , 0. ],
           [0. , 0. , 0. , ..., 0. , 1. , 0. ],
           [0. , 0. , 0. , ..., 0. , 0. , 1. ]])




```python
indices['Avatar'] #영화가 몇번째 index에 해당하는지 확인해 준다
```




    0




```python
df2 = df2.reset_index()
indices = pd.Series(df2.index, index=df2['title'])
indices
```




    title
    Avatar                                         0
    Pirates of the Caribbean: At World's End       1
    Spectre                                        2
    The Dark Knight Rises                          3
    John Carter                                    4
                                                ... 
    El Mariachi                                 4798
    Newlyweds                                   4799
    Signed, Sealed, Delivered                   4800
    Shanghai Calling                            4801
    My Date with Drew                           4802
    Length: 4803, dtype: int64




```python
get_recommendations('The Dark Knight Rises', cosine_sim2)
```




    65               The Dark Knight
    119                Batman Begins
    4638    Amidst the Devil's Wings
    1196                The Prestige
    3073           Romeo Is Bleeding
    3326              Black November
    1503                      Takers
    1986                      Faster
    303                     Catwoman
    747               Gangster Squad
    Name: title, dtype: object




```python
get_recommendations('Up', cosine_sim2) #Up과 유사한 영화 추천
```




    231                                        Monsters, Inc.
    1983                                     Meet the Deedles
    3403    Alpha and Omega: The Legend of the Saw Tooth Cave
    3114                                          Elsa & Fred
    1580                                          The Nut Job
    3670                                      Running Forever
    4709                            A Charlie Brown Christmas
    40                                                 Cars 2
    42                                            Toy Story 3
    77                                             Inside Out
    Name: title, dtype: object




```python
get_recommendations('The Martian', cosine_sim2) #The Martian과 유사한 영화 추천
```




    4                    John Carter
    95                  Interstellar
    365                      Contact
    256                    Allegiant
    1326                The 5th Wave
    1958                 On the Road
    3043            End of the Spear
    3373    The Other Side of Heaven
    3392                       Gerry
    3698                   Moby Dick
    Name: title, dtype: object




```python
indices['The Martian'] #The Martian에대한 index 값
```




    270




```python
df2.loc[270] # 마션 영화 데이터
```




    index                                                                 270
    budget                                                          108000000
    genres                                 [drama, adventure, sciencefiction]
    homepage                      http://www.foxmovies.com/movies/the-martian
    id                                                                 286217
    keywords                                       [basedonnovel, mars, nasa]
    original_language                                                      en
    original_title                                                The Martian
    overview                During a manned mission to Mars, Astronaut Mar...
    popularity                                                      167.93287
    production_companies    [{"name": "Twentieth Century Fox Film Corporat...
    production_countries    [{"iso_3166_1": "US", "name": "United States o...
    release_date                                                   2015-09-30
    revenue                                                         630161890
    runtime                                                             141.0
    spoken_languages        [{"iso_639_1": "en", "name": "English"}, {"iso...
    status                                                           Released
    tagline                                                    Bring Him Home
    title                                                         The Martian
    vote_average                                                          7.6
    vote_count                                                           7268
    cast                            [mattdamon, jessicachastain, kristenwiig]
    crew                    [{'credit_id': '5607a7e19251413050003e2c', 'de...
    director                                                      ridleyscott
    soup                    basedonnovel mars nasa mattdamon jessicachasta...
    Name: 270, dtype: object




```python
df2.loc[4]
```




    index                                                                   4
    budget                                                          260000000
    genres                                [action, adventure, sciencefiction]
    homepage                             http://movies.disney.com/john-carter
    id                                                                  49529
    keywords                                  [basedonnovel, mars, medallion]
    original_language                                                      en
    original_title                                                John Carter
    overview                John Carter is a war-weary, former military ca...
    popularity                                                      43.926995
    production_companies          [{"name": "Walt Disney Pictures", "id": 2}]
    production_countries    [{"iso_3166_1": "US", "name": "United States o...
    release_date                                                   2012-03-07
    revenue                                                         284139100
    runtime                                                             132.0
    spoken_languages                 [{"iso_639_1": "en", "name": "English"}]
    status                                                           Released
    tagline                              Lost in our world, found in another.
    title                                                         John Carter
    vote_average                                                          6.1
    vote_count                                                           2124
    cast                          [taylorkitsch, lynncollins, samanthamorton]
    crew                    [{'credit_id': '52fe479ac3a36847f813eaa3', 'de...
    director                                                    andrewstanton
    soup                    basedonnovel mars medallion taylorkitsch lynnc...
    Name: 4, dtype: object




```python
get_recommendations('The Avengers', cosine_sim2)
```




    7                  Avengers: Age of Ultron
    26              Captain America: Civil War
    79                              Iron Man 2
    169     Captain America: The First Avenger
    174                    The Incredible Hulk
    85     Captain America: The Winter Soldier
    31                              Iron Man 3
    33                   X-Men: The Last Stand
    68                                Iron Man
    94                 Guardians of the Galaxy
    Name: title, dtype: object



 웹사이트로 만들어 보기(포트폴리오 처럼)


```python
import pickle #파일에 저장하거나, 네트워크에 전송하기 위한 라이브러리
```


```python
df2.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>keywords</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>popularity</th>
      <th>...</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>cast</th>
      <th>crew</th>
      <th>director</th>
      <th>soup</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>237000000</td>
      <td>[action, adventure, fantasy]</td>
      <td>http://www.avatarmovie.com/</td>
      <td>19995</td>
      <td>[cultureclash, future, spacewar]</td>
      <td>en</td>
      <td>Avatar</td>
      <td>In the 22nd century, a paraplegic Marine is di...</td>
      <td>150.437577</td>
      <td>...</td>
      <td>[{"iso_639_1": "en", "name": "English"}, {"iso...</td>
      <td>Released</td>
      <td>Enter the World of Pandora.</td>
      <td>Avatar</td>
      <td>7.2</td>
      <td>11800</td>
      <td>[samworthington, zoesaldana, sigourneyweaver]</td>
      <td>[{'credit_id': '52fe48009251416c750aca23', 'de...</td>
      <td>jamescameron</td>
      <td>cultureclash future spacewar samworthington zo...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>300000000</td>
      <td>[adventure, fantasy, action]</td>
      <td>http://disney.go.com/disneypictures/pirates/</td>
      <td>285</td>
      <td>[ocean, drugabuse, exoticisland]</td>
      <td>en</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>Captain Barbossa, long believed to be dead, ha...</td>
      <td>139.082615</td>
      <td>...</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>At the end of the world, the adventure begins.</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>6.9</td>
      <td>4500</td>
      <td>[johnnydepp, orlandobloom, keiraknightley]</td>
      <td>[{'credit_id': '52fe4232c3a36847f800b579', 'de...</td>
      <td>goreverbinski</td>
      <td>ocean drugabuse exoticisland johnnydepp orland...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>245000000</td>
      <td>[action, adventure, crime]</td>
      <td>http://www.sonypictures.com/movies/spectre/</td>
      <td>206647</td>
      <td>[spy, basedonnovel, secretagent]</td>
      <td>en</td>
      <td>Spectre</td>
      <td>A cryptic message from Bond’s past sends him o...</td>
      <td>107.376788</td>
      <td>...</td>
      <td>[{"iso_639_1": "fr", "name": "Fran\u00e7ais"},...</td>
      <td>Released</td>
      <td>A Plan No One Escapes</td>
      <td>Spectre</td>
      <td>6.3</td>
      <td>4466</td>
      <td>[danielcraig, christophwaltz, léaseydoux]</td>
      <td>[{'credit_id': '54805967c3a36829b5002c41', 'de...</td>
      <td>sammendes</td>
      <td>spy basedonnovel secretagent danielcraig chris...</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 25 columns</p>
</div>




```python
movies = df2[['id', 'title']].copy()
movies.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19995</td>
      <td>Avatar</td>
    </tr>
    <tr>
      <th>1</th>
      <td>285</td>
      <td>Pirates of the Caribbean: At World's End</td>
    </tr>
    <tr>
      <th>2</th>
      <td>206647</td>
      <td>Spectre</td>
    </tr>
    <tr>
      <th>3</th>
      <td>49026</td>
      <td>The Dark Knight Rises</td>
    </tr>
    <tr>
      <th>4</th>
      <td>49529</td>
      <td>John Carter</td>
    </tr>
  </tbody>
</table>
</div>




```python
pickle.dump(movies, open('movies.pickle', 'wb'))
```


```python
pickle.dump(cosine_sim2, open('cosine_sim.pickle', 'wb'))
```
