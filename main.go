package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"time"
)

/*
使用屏幕共享，完成如下小任务。请当成工作场景，你可以查阅任何资料，使用任何工具，没有任何限制：

```
curl https://rest.coinapi.io/v1/exchangerate/BTC/USD --header "X-CoinAPI-Key: B89898B1-1DFC-4D44-AB49-4D56856A3627"
```
这是一个数字货币汇率的第三方API，path中有一对数字货币的符号，比如现在是 BTC 的美元价格。

1. 在golang中请求这个 API，设计一个 struct，把结果存入其中，并打印出来。
2. 实现一个 http 服务器，框架不限，简化这个API功能为：用任何一种数字货币符号当做参数，返回它的美元价格。URL格式自定。
3. 为了更快的响应，和节约费用，为这个 API 增加一个缓存（在内存实现就好），可以让已经请求过的结果在 10秒内不用再调用上游 API 。

*/

const XCoinAPIKey = "B89898B1-1DFC-4D44-AB49-4D56856A3627"
const TokenPriceUrlPrefix = "https://rest.coinapi.io/v1/exchangerate"

type TokenPrice struct {
	Time  time.Time `json:"time"`
	Token string    `json:"asset_id_base"`
	Quote string    `json:"asset_id_quote"`
	Rate  float64   `json:"rate"`
}

var localCache *cache.Cache

func main() {
	// Create a cache with a default expiration time of 5 minutes, and which
	// purges expired items every 10 minutes
	localCache = cache.New(5*time.Minute, 10*time.Minute)

	val, err := getTokenUSDPrice("BTC")
	fmt.Println(val, err)
}

func getTokenUSDPrice(token string) (*TokenPrice, error) {
	// Combine url
	fullURL := fmt.Sprintf("%s/%s/%s", TokenPriceUrlPrefix, token, "USD")

	// Get the string associated with the key token from the cache
	cachedVal, found := localCache.Get(token)
	if found {
		ret := &TokenPrice{}
		if err := json.Unmarshal([]byte(cachedVal.(string)), ret); err != nil {
			return nil, err
		}
		return ret, nil
	}

	// if not in cache, call http request
	tokenPrice, err := sendAndGetHttpRequest(fullURL, map[string]string{"X-CoinAPI-Key": XCoinAPIKey})
	if err != nil {
		return nil, err
	}
	tpb, err := json.Marshal(tokenPrice)
	if err != nil {
		return nil, err
	}
	localCache.Set(token, string(tpb), time.Second*10) // set cache
	return &tokenPrice, err
}

func sendAndGetHttpRequest(url string, headers map[string]string) (tp TokenPrice, err error) {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return
	}

	for key, val := range headers {
		req.Header.Add(key, val)
	}

	// 创建 HTTP 客户端并发送请求
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return
	}
	defer resp.Body.Close()

	// 读取响应内容
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		fmt.Printf("HTTP GET error: %v\n", err)
		return
	}

	ret := TokenPrice{}
	err = json.Unmarshal(body, &ret)
	if err != nil {
		return
	}
	return ret, nil
}
